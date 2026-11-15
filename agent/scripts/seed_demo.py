"""Seed the HR chatbot demo dataset through the Odoo MCP endpoint.

Run from the repository root:
    uv run python scripts/seed_demo.py

The script is idempotent: existing records are reused and every MCP call is
throttled to stay below the default 60-request-per-minute key limit.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ont_agent.config import load_settings  # noqa: E402
from ont_agent.tools import get_odoo_tools  # noqa: E402

COMPANY_NAME = "Acme India"
DEMO_YEAR = 2026
CALL_DELAY_SECONDS = 3.5
DESIGNATION = "Junior Software programmer"
MONTHLY_WAGE = 18_000
ANNUAL_LEAVE_DAYS = 10

EMPLOYEES = [
    ("John Mathew", "Alwin hemanth", "Kochi", "682001", "1990-05-14"),
    ("Anita Varma", "Abin varghese", "Kochi", "682016", "1988-11-03"),
    ("Rahul Nair", "Adithyan Kg", "Aluva", "683101", "1995-07-22"),
    ("Meera Pillai", "Sanjay P", "Aluva", "683102", "1994-02-18"),
    ("Vishnu Das", "Ajeesh P", "Kochi", "682024", "1985-09-09"),
    ("Divya Menon", "Amarnath", "Thiruvananthapuram", "695001", "1992-04-27"),
    ("Arjun Krishnan", "Anusree", "Chennai", "600001", "2000-12-08"),
    ("Lakshmi Suresh", "Navya vijayakrishnan", "Thiruvananthapuram", "695010", "1991-06-16"),
]

CONTRACTS = {
    "Anita Varma": [
        ("2024-01-01", "2024-12-31", MONTHLY_WAGE, "close"),
        ("2025-01-01", "2025-12-31", MONTHLY_WAGE, "close"),
        ("2026-01-01", None, MONTHLY_WAGE, "open"),
    ],
    "John Mathew": [
        ("2024-06-01", "2025-05-31", MONTHLY_WAGE, "close"),
        ("2025-06-01", None, MONTHLY_WAGE, "open"),
    ],
    "Rahul Nair": [
        ("2025-03-01", "2025-12-31", MONTHLY_WAGE, "close"),
        ("2026-01-01", None, MONTHLY_WAGE, "open"),
    ],
    "Meera Pillai": [("2025-01-01", None, MONTHLY_WAGE, "open")],
    "Vishnu Das": [("2024-01-01", None, MONTHLY_WAGE, "open")],
    "Divya Menon": [("2025-06-01", None, MONTHLY_WAGE, "open")],
    "Arjun Krishnan": [("2026-02-01", None, MONTHLY_WAGE, "open")],
    "Lakshmi Suresh": [("2024-09-01", None, MONTHLY_WAGE, "open")],
}

SKILLS = {
    "Vishnu Das": ("Expert", "Java Architect / Developer", "2016-01-01", None),
    "John Mathew": ("Advanced", "Java Developer at Acme", "2019-05-01", "2024-04-30"),
    "Rahul Nair": ("Intermediate", "Java Backend Developer", "2021-06-01", None),
    "Arjun Krishnan": ("Beginner", "Java Trainee / Junior Developer", "2023-07-01", None),
}

LEGACY_NOTES = {
    "Anita Varma": [
        "Salary increase: 50,000 to 58,000 effective Jan 2025 (annual revision).",
        "Salary increase: 58,000 to 65,000 effective Jan 2026 (promotion).",
        "Incentive Jun 2025: INR 20,000 - Project Phoenix delivery.",
        "Bonus Dec 2025: INR 50,000.",
    ],
    "John Mathew": [
        "Salary increase: 60,000 to 72,000 effective Jun 2025.",
        "Bonus Dec 2025: INR 40,000.",
    ],
    "Rahul Nair": ["Bonus Dec 2025: INR 30,000."],
    "Vishnu Das": ["Bonus Dec 2025: INR 60,000."],
}


class Seeder:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}
        self._last_call = 0.0

    async def connect(self) -> None:
        settings = load_settings(ROOT / ".env")
        tools = await get_odoo_tools(settings.mcp_url, settings.mcp_bearer_key)
        self.tools = {tool.name: tool for tool in tools}
        required = {
            "search_read",
            "create_records",
            "delete_records",
            "update_records",
            "post_message",
            "call_method",
        }
        missing = required - self.tools.keys()
        if missing:
            raise RuntimeError(f"MCP tools missing: {', '.join(sorted(missing))}")

    async def call(self, name: str, **arguments: Any) -> Any:
        delay = CALL_DELAY_SECONDS - (time.monotonic() - self._last_call)
        if delay > 0:
            await asyncio.sleep(delay)
        for attempt in range(6):
            try:
                result = await self.tools[name].ainvoke(arguments)
                self._last_call = time.monotonic()
                parsed = self._parse(result)
                if "rate limit exceeded" not in str(parsed).lower():
                    return parsed
            except Exception as exc:
                safe_disconnected_call = name in {"search_read", "update_records"} or (
                    name == "call_method" and arguments.get("method") == "action_validate"
                )
                disconnected_read = (
                    safe_disconnected_call and exc.__class__.__name__ == "BrokenResourceError"
                )
                if not (self._is_rate_limit_error(exc) or disconnected_read) or attempt == 5:
                    raise
            await asyncio.sleep(65)
        raise RuntimeError(f"MCP call {name} remained rate limited")

    @classmethod
    def _is_rate_limit_error(cls, error: BaseException) -> bool:
        message = str(error).lower()
        if "429" in message or "rate limit" in message:
            return True
        if isinstance(error, BaseExceptionGroup):
            return any(cls._is_rate_limit_error(item) for item in error.exceptions)
        return False

    @staticmethod
    def _parse(result: Any) -> Any:
        if isinstance(result, tuple) and result:
            result = result[0]
        if isinstance(result, list):
            parts = []
            for item in result:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    text = getattr(item, "text", None)
                    parts.append(str(text) if text is not None else str(item))
            text = "\n".join(parts).strip()
        elif isinstance(result, str):
            text = result.strip()
        else:
            return result
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            if text.lower().startswith(("error", "access denied")):
                raise RuntimeError(text) from exc
            return text

    async def search(
        self, model: str, domain: list[list[Any]], fields: list[str]
    ) -> list[dict[str, Any]]:
        result = await self.call(
            "search_read",
            model=model,
            domain=domain,
            fields=fields,
            limit=100,
        )
        if not isinstance(result, list):
            raise RuntimeError(f"Unexpected search_read result for {model}: {result!r}")
        return result

    async def one(
        self, model: str, domain: list[list[Any]], fields: list[str]
    ) -> dict[str, Any] | None:
        rows = await self.search(model, domain, fields)
        return rows[0] if rows else None

    async def ensure(
        self, model: str, domain: list[list[Any]], values: dict[str, Any]
    ) -> dict[str, Any]:
        existing = await self.one(model, domain, ["id", "display_name"])
        if existing:
            print(f"= {model}: {existing.get('display_name', existing['id'])}")
            return existing
        try:
            created = await self.call("create_records", model=model, values=values)
        except Exception as exc:
            if exc.__class__.__name__ != "BrokenResourceError":
                raise
            await asyncio.sleep(65)
            existing = await self.one(model, domain, ["id", "display_name"])
            if existing:
                print(f"= {model}: {existing.get('display_name', existing['id'])}")
                return existing
            created = await self.call("create_records", model=model, values=values)
        if not isinstance(created, dict) or "id" not in created:
            raise RuntimeError(f"Could not create {model}: {created!r}")
        print(f"+ {model}: {created.get('display_name', created['id'])}")
        return created

    async def ensure_note(self, employee_id: int, body: str) -> None:
        existing = await self.one(
            "mail.message",
            [["model", "=", "hr.employee"], ["res_id", "=", employee_id], ["body", "ilike", body]],
            ["id"],
        )
        if existing:
            print(f"= note: {body}")
            return
        try:
            await self.call(
                "post_message", model="hr.employee", id=employee_id, body=body, type="note"
            )
        except Exception as exc:
            if exc.__class__.__name__ != "BrokenResourceError":
                raise
            await asyncio.sleep(65)
            existing = await self.one(
                "mail.message",
                [
                    ["model", "=", "hr.employee"],
                    ["res_id", "=", employee_id],
                    ["body", "ilike", body],
                ],
                ["id"],
            )
            if existing:
                print(f"= note: {body}")
                return
            await self.call(
                "post_message", model="hr.employee", id=employee_id, body=body, type="note"
            )
        print(f"+ note: {body}")


async def seed() -> None:
    seeder = Seeder()
    await seeder.connect()

    company = await seeder.one("res.company", [["name", "=", COMPANY_NAME]], ["id", "name"])
    if not company:
        raise RuntimeError(f"Company {COMPANY_NAME!r} was not found")
    company_id = company["id"]
    print(f"Target company: {company.get('display_name', COMPANY_NAME)} (#{company_id})")

    personal = await seeder.ensure(
        "hr.employee.category",
        [["name", "=", f"Requested Personal Growth {DEMO_YEAR}"]],
        {"name": f"Requested Personal Growth {DEMO_YEAR}", "color": 4},
    )
    suggested = await seeder.ensure(
        "hr.employee.category",
        [["name", "=", f"Management Suggested Course {DEMO_YEAR}"]],
        {"name": f"Management Suggested Course {DEMO_YEAR}", "color": 2},
    )

    personal_names = {"John Mathew", "Meera Pillai", "Divya Menon"}
    suggested_names = {"Rahul Nair", "Arjun Krishnan"}
    employees: dict[str, int] = {}
    display_names: dict[str, str] = {}
    for index, (legacy_name, name, city, zipcode, birthday) in enumerate(EMPLOYEES, start=1):
        tags = []
        if legacy_name in personal_names:
            tags.append(personal["id"])
        if legacy_name in suggested_names:
            tags.append(suggested["id"])
        values = {
            "name": name,
            "job_title": DESIGNATION,
            "work_email": f"demo.employee{index}@example.invalid",
            "birthday": birthday,
            "private_city": city,
            "private_zip": zipcode,
            "company_id": company_id,
            "category_ids": [[6, 0, tags]],
        }
        employee = await seeder.one(
            "hr.employee",
            [["work_email", "=", values["work_email"]], ["company_id", "=", company_id]],
            ["id", "display_name"],
        )
        if employee is None:
            employee = await seeder.one(
                "hr.employee",
                [
                    ["name", "in", [name, legacy_name]],
                    ["company_id", "=", company_id],
                ],
                ["id", "display_name"],
            )
        if employee is None:
            employee = await seeder.ensure(
                "hr.employee",
                [["name", "=", name], ["company_id", "=", company_id]],
                values,
            )
        await seeder.call(
            "update_records",
            model="hr.employee",
            ids=[employee["id"]],
            values=values,
        )
        employees[legacy_name] = employee["id"]
        display_names[legacy_name] = name

    for name, contracts in CONTRACTS.items():
        for start, end, wage, state in contracts:
            values: dict[str, Any] = {
                "name": f"DEMO {name} - {start}",
                "employee_id": employees[name],
                "company_id": company_id,
                "date_start": start,
                "wage": wage,
                "state": state,
            }
            if end:
                values["date_end"] = end
            contract = await seeder.ensure(
                "hr.contract",
                [["employee_id", "=", employees[name]], ["date_start", "=", start]],
                values,
            )
            await seeder.call(
                "update_records",
                model="hr.contract",
                ids=[contract["id"]],
                values={
                    "name": f"DEMO {display_names[name]} - {start}",
                    "wage": MONTHLY_WAGE,
                },
            )

    leave_type_values = {
        "name": f"Annual Leave {DEMO_YEAR}",
        "company_id": company_id,
        "requires_allocation": "yes",
        "leave_validation_type": "no_validation",
        "allocation_validation_type": "no",
    }
    leave_type = await seeder.one(
        "hr.leave.type",
        [["name", "=", f"Annual Leave {DEMO_YEAR}"], ["company_id", "=", company_id]],
        ["id", "display_name"],
    )
    if leave_type is None:
        leave_type = await seeder.one(
            "hr.leave.type",
            [
                ["name", "=", f"Demo Paid Time Off {DEMO_YEAR}"],
                ["company_id", "=", company_id],
            ],
            ["id", "display_name"],
        )
    if leave_type is None:
        leave_type = await seeder.ensure(
            "hr.leave.type",
            [["name", "=", f"Annual Leave {DEMO_YEAR}"], ["company_id", "=", company_id]],
            leave_type_values,
        )
    else:
        await seeder.call(
            "update_records",
            model="hr.leave.type",
            ids=[leave_type["id"]],
            values=leave_type_values,
        )

    for legacy_name, name, *_details in EMPLOYEES:
        employee_id = employees[legacy_name]
        old_leaves = await seeder.search(
            "hr.leave",
            [["employee_id", "=", employee_id], ["name", "ilike", f"DEMO {DEMO_YEAR} PTO"]],
            ["id", "state"],
        )
        for leave in old_leaves:
            if leave.get("state") == "validate":
                await seeder.call(
                    "call_method",
                    model="hr.leave",
                    method="action_refuse",
                    ids=[leave["id"]],
                )
            await seeder.call("delete_records", model="hr.leave", ids=[leave["id"]])

        allocation_name = f"Annual Leave {DEMO_YEAR} - {name}"
        allocation = await seeder.one(
            "hr.leave.allocation",
            [
                ["employee_id", "=", employee_id],
                ["holiday_status_id", "=", leave_type["id"]],
                ["date_from", "=", f"{DEMO_YEAR}-01-01"],
            ],
            ["id", "state"],
        )
        allocation_values = {
            "name": allocation_name,
            "employee_id": employee_id,
            "holiday_status_id": leave_type["id"],
            "number_of_days": ANNUAL_LEAVE_DAYS,
            "date_from": f"{DEMO_YEAR}-01-01",
        }
        if allocation is None:
            allocation = await seeder.ensure(
                "hr.leave.allocation",
                [["name", "=", allocation_name], ["employee_id", "=", employee_id]],
                allocation_values,
            )
        else:
            await seeder.call(
                "update_records",
                model="hr.leave.allocation",
                ids=[allocation["id"]],
                values=allocation_values,
            )
        if allocation.get("state") != "validate":
            await seeder.call(
                "call_method",
                model="hr.leave.allocation",
                method="action_validate",
                ids=[allocation["id"]],
            )

    skill_type = await seeder.ensure(
        "hr.skill.type",
        [["name", "=", "Programming"]],
        {"name": "Programming"},
    )
    levels: dict[str, int] = {}
    for name, progress in [
        ("Beginner", 25),
        ("Intermediate", 50),
        ("Advanced", 75),
        ("Expert", 100),
    ]:
        level = await seeder.ensure(
            "hr.skill.level",
            [["name", "=", name], ["skill_type_id", "=", skill_type["id"]]],
            {
                "name": name,
                "skill_type_id": skill_type["id"],
                "level_progress": progress,
            },
        )
        levels[name] = level["id"]
    java = await seeder.ensure(
        "hr.skill",
        [["name", "=", "Java"], ["skill_type_id", "=", skill_type["id"]]],
        {"name": "Java", "skill_type_id": skill_type["id"]},
    )
    experience_type = await seeder.ensure(
        "hr.resume.line.type",
        [["name", "=", "Experience"]],
        {"name": "Experience", "sequence": 10},
    )
    for name, (level_name, role, start, end) in SKILLS.items():
        await seeder.ensure(
            "hr.employee.skill",
            [["employee_id", "=", employees[name]], ["skill_id", "=", java["id"]]],
            {
                "employee_id": employees[name],
                "skill_id": java["id"],
                "skill_type_id": skill_type["id"],
                "skill_level_id": levels[level_name],
            },
        )
        resume_values: dict[str, Any] = {
            "name": role,
            "employee_id": employees[name],
            "line_type_id": experience_type["id"],
            "date_start": start,
        }
        if end:
            resume_values["date_end"] = end
        await seeder.ensure(
            "hr.resume.line",
            [["employee_id", "=", employees[name]], ["name", "=", role]],
            resume_values,
        )

    for name, notes in LEGACY_NOTES.items():
        for note in notes:
            messages = await seeder.search(
                "mail.message",
                [
                    ["model", "=", "hr.employee"],
                    ["res_id", "=", employees[name]],
                    ["body", "ilike", note],
                ],
                ["id"],
            )
            if messages:
                await seeder.call(
                    "delete_records",
                    model="mail.message",
                    ids=[message["id"] for message in messages],
                )
    for employee_id in employees.values():
        await seeder.ensure_note(employee_id, "Monthly salary: INR 18,000.")

    print("\nDemo HR dataset is ready. Questions 3, 5, 6 and 8 remain live-write demos.")


def main() -> None:
    asyncio.run(seed())


if __name__ == "__main__":
    main()
