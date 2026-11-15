"""Thread rotation: bound a conversation that never goes idle.

Retention (``housekeeping.py``) deletes threads nobody has spoken to in a
month. It cannot help the user it was written for — someone who chats
daily is never idle, so their single thread grows forever at roughly sixty
checkpoints a day, each a full snapshot of conversation state. One chat
reached 98 MB before any of this existed.

Rotation caps that: past a threshold the chat moves to a FRESH thread,
carrying the conversation across, and the old one is deleted. Nothing is
announced, because nothing is lost — see ``agent.thread_messages`` for why
carrying ``messages`` carries everything, and why the checkpoint history
left behind is dead weight.

**Two phases, and the split is the whole safety argument.**
``plan_rotation`` only decides; ``Rotation.commit`` runs after the turn has
returned. Until it does, the epoch is unchanged and the old thread is
intact, so a crash anywhere in between costs nothing: the next turn simply
rotates again, and the half-written new thread is an orphan the 30-day
retention pass collects. The old thread is deleted only once its content
is provably somewhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from ont_agent.agent import AgentGraph, has_pending, thread_messages
from ont_agent.config import Settings
from ont_agent.log import get_logger
from ont_agent.store import BindingStore, binding_id

logger = get_logger(__name__)


def _epoch_key(chat_id: int) -> str:
    """Where a chat's epoch lives in the existing ``meta`` table.

    Not a ``bindings`` column, which would be the natural home: ``Binding``
    is a frozen dataclass rebuilt field by field in
    ``auth.py:_refresh_binding``, where five fields are already carried by
    hand. Forgetting a sixth would reset the epoch on the next token
    refresh and silently resurrect a thread id that had been deleted.
    """
    return f"thread_epoch:{chat_id}"


def thread_name(chat_id: int, epoch: int) -> str:
    """Epoch 0 keeps the bare chat id, so existing threads survive this change."""
    return str(chat_id) if epoch <= 0 else f"{chat_id}-{epoch}"


async def _read_epoch(store: BindingStore, chat_id: int) -> int:
    raw = await store.get_meta(_epoch_key(chat_id))
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        # Benign: epoch 0's thread was deleted at the first rotation, so the
        # chat starts fresh exactly as /new would rather than reading someone
        # else's state.
        logger.warning("Unreadable thread epoch %r for chat %s", raw, binding_id(chat_id))
        return 0


async def current_thread(store: BindingStore, chat_id: int) -> str:
    """The thread this chat is on right now — for callers that must not rotate.

    ``resume_pending`` is the reason this exists: an approval is mid-flight,
    and the thread holding it is the only one that can accept the decision.
    """
    return thread_name(chat_id, await _read_epoch(store, chat_id))


@dataclass
class Rotation:
    """The thread a turn should use, and how to make that permanent."""

    #: The thread this turn must run on — new if rotating, current if not.
    thread_id: str
    #: Messages to seed the new thread with. Empty unless rotating.
    history: list[Any] = field(default_factory=list)
    #: The thread being retired. None when this turn is not a rotation.
    previous_thread: str | None = None
    #: Epoch to record on commit.
    epoch: int = 0

    @property
    def rotated(self) -> bool:
        return self.previous_thread is not None

    async def commit(
        self,
        store: BindingStore,
        memory: BaseCheckpointSaver[Any] | None,
        chat_id: int,
    ) -> None:
        """Make the new thread the current one, then retire the old.

        Called once the turn has returned — including a turn that ended in a
        pending approval, whose decision ``resume_pending`` must find on the
        NEW thread.

        Never raises. Maintenance must not turn a delivered answer into an
        error, and every failure here is recoverable on its own: an epoch
        that was not written means the next turn rotates again, and a thread
        that was not deleted ages out through retention.
        """
        if self.previous_thread is None:
            return
        try:
            await store.set_meta(_epoch_key(chat_id), str(self.epoch))
        except Exception:
            # The old thread stays current, so this turn's answer is stranded
            # on a thread nobody reads — one turn of context, not the chat.
            logger.error(
                "Could not record rotation for chat %s — keeping the old thread",
                binding_id(chat_id),
                exc_info=True,
            )
            return
        logger.info(
            "Chat %s: rotated conversation %s -> %s (%d message(s) carried)",
            binding_id(chat_id),
            self.previous_thread,
            self.thread_id,
            len(self.history),
        )
        if memory is None:
            return
        try:
            await memory.adelete_thread(self.previous_thread)
        except Exception:
            # Safe to leave: it is now an orphan, and retention collects it.
            logger.error("Could not delete rotated thread %s", self.previous_thread, exc_info=True)


async def plan_rotation(
    store: BindingStore,
    agent: AgentGraph,
    settings: Settings,
    chat_id: int,
    memory: BaseCheckpointSaver[Any] | None,
) -> Rotation:
    """Decide which thread this turn runs on, rotating if the current one is full.

    Nothing is written here. The caller runs the turn on ``thread_id`` with
    ``history`` prepended, then calls ``commit``.
    """
    epoch = await _read_epoch(store, chat_id)
    staying = Rotation(thread_id=thread_name(chat_id, epoch), epoch=epoch)
    limit = settings.thread_max_checkpoints
    if memory is None or limit <= 0:
        return staying
    try:
        if await store.count_checkpoints(staying.thread_id) < limit:
            return staying
        if await has_pending(agent, staying.thread_id):
            # The paused approval lives in this thread's checkpoint, and
            # there is no way to carry an interrupt across. It rotates on
            # the next turn instead, once the decision has settled.
            logger.info(
                "Chat %s: rotation deferred, an approval is still open", binding_id(chat_id)
            )
            return staying
        history = await thread_messages(agent, staying.thread_id)
    except Exception:
        # Rotation is maintenance: failing it costs disk, and raising here
        # would break the question the user actually asked.
        logger.error("Could not plan rotation for chat %s", binding_id(chat_id), exc_info=True)
        return staying
    return Rotation(
        thread_id=thread_name(chat_id, epoch + 1),
        history=history,
        previous_thread=staying.thread_id,
        epoch=epoch + 1,
    )
