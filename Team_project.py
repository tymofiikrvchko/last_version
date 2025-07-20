import os
import datetime
import pickle
import re
from collections import UserDict
from typing import Optional, List, Tuple, Type

from rich.columns import Columns
# ────────────────────────────────────────────────────────────────────────────
# Rich console
# ────────────────────────────────────────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ────────────────────────────────────────────────────────────────────────────
# Registration
# ────────────────────────────────────────────────────────────────────────────
import getpass

console = Console()

# ────────────────────────────────────────────────────────────────────────────
# Optional OpenAI (autocorrect & semantic search); can work without key.txt
# ────────────────────────────────────────────────────────────────────────────
try:
    from openai import OpenAI

    with open("key.txt", "r", encoding="utf‑8") as f:
        _client = OpenAI(api_key=f.read().strip())
except (ImportError, FileNotFoundError):
    _client = None

# ────────────────────────────────────────────────────────────────────────────
# Command dictionaries – кратко; полный список по команде help внутри режима
# ────────────────────────────────────────────────────────────────────────────
CONTACT_DESC = {
    "add": "add <Name> [Surname] [Phone] [Email] [Address]",
    "change": "change <Name> – Заменить номер телефона или Email или адрес ",
    "remove-phone": "remove-phone <Name> <Phone>",
    "phone": "phone <Name>",
    "delete": "delete <Name>",
    "all": "all – показать все контакты",
    "search": "search <query> – имя/фамилия/телефон/заметки",
    "add-birthday": "add-birthday <Name> <DD.MM.YYYY>",
    "show-birthday": "show-birthday <Name|Surname>",
    "birthdays": "birthdays <N> – ближайшие N дней",
    "add-contact-note": "add-contact-note <Name> <Text>",
    "change-address": "change-address <Name> <New address>",
    "change-email": "change-email <Name> <New email>",
    "back": "back – главное меню",
    "exit": "exit / close – сохранить и выйти",
    "hello": "hello / help – показать помощь",
    "help": "help – то же самое",
}

NOTE_DESC = {
    "add-note": "add new note",
    "list-notes": "view all notes",
    "add-tag": "add new tags",
    "search-tag": "find a note by tag",
    "search-note": "find note by text",
    "back": "return to mode selection",
    "exit | close": "end assistant work",
    "hello | help": "output all commands"
}


# ────────────────────────────────────────────────────────────────────────────
# GPT‑autocorrect helper
# ────────────────────────────────────────────────────────────────────────────
def suggest_correction(user_input: str,
                       desc_map: dict[str, str]) -> Optional[str]:
    """
    Просим GPT‑4o‑mini угадать опечатанную команду.
    Возвращает canonical‑имя команды или None.
    """
    if _client is None:
        return None
    sys_prompt = (
            "You are a CLI assistant that fixes mistyped commands. "
            "User may write RU/UA/EN with typos.\n\n"
            "Supported commands:\n" +
            "\n".join(desc_map.keys()) +
            "\n\nReturn ONLY the canonical command name or empty string."
    )
    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.0,
        max_tokens=6
    )
    guess = resp.choices[0].message.content.strip().strip("\"'")
    return guess if guess in desc_map else None


# ────────────────────────────────────────────────────────────────────────────
# Data model
# ────────────────────────────────────────────────────────────────────────────
class Field:
    def __init__(self, value):  self.value = value

    def __str__(self):          return str(self.value)


class Name(Field):
    def __init__(self, value: str):
        if not value.strip():
            raise ValueError("Name cannot be empty.")
        super().__init__(value.strip())


class Surname(Field):  pass


class Address(Field):  pass


class Email(Field):
    EMAIL_RE = re.compile(r"[^@]+@[^@]+\.[^@]+")

    def __init__(self, value: str):
        v = value.strip()
        if v and not Email.EMAIL_RE.fullmatch(v):
            raise ValueError("Invalid e‑mail format.")
        super().__init__(v)


class Phone(Field):
    def __init__(self, value: str):
        if not value.isdigit() or len(value) != 10:
            raise ValueError("Phone must contain exactly 10 digits.")
        super().__init__(value)


class Birthday(Field):
    def __init__(self, value: str):
        try:
            dt = datetime.datetime.strptime(value, "%d.%m.%Y").date()
        except ValueError:
            raise ValueError("Date must be DD.MM.YYYY")
        if dt > datetime.date.today():
            raise ValueError("Birthday cannot be in the future.")
        super().__init__(dt)


class Record:
    def __init__(self, name: str, surname: str = "", address: str = "", email: str = ""):
        self.name = Name(name)
        self.surname = Surname(surname)
        self.address = address if isinstance(address, Address) else Address(address)
        self.email = email if isinstance(email, Email) else Email(email)
        self.phones: List[Phone] = []
        self.birthday: Optional[Birthday] = None
        self.contact_notes: List[str] = []

    # phone ops
    def add_phone(self, phone: str):
        self.phones.append(Phone(phone))

    def remove_phone(self, phone: str):
        self.phones = [p for p in self.phones if p.value != phone]

    def edit_phone(self, idx: int, new: str):
        self.phones[idx] = Phone(new)

    # misc
    def add_birthday(self, date_str: str):
        if self.birthday:
            raise ValueError("Birthday already set.")
        self.birthday = Birthday(date_str)

    def add_contact_note(self, note: str):
        if not note.strip():
            raise ValueError("Note cannot be empty.")
        self.contact_notes.append(note.strip())

    def update_email(self, email: str):
        self.email = Email(email)

    def update_address(self, addr: str):
        self.address = Address(addr)


class AddressBook(UserDict):
    def add_record(self, rec: Record):
        key = make_key(rec.name.value, rec.surname.value)
        self.data[key] = rec

    def find(self, name: str) -> Record:
        key = get_record_key(name, self)
        if key is None:
            raise KeyError("Contact not found.")
        return self.data[key]

    def delete(self, name: str):
        del self.data[make_key_from_input(name)]

    # --- главное: ближайшие ДР ---
    def upcoming(self, days_ahead: int) -> dict[str, Tuple[datetime.date, int]]:
        """
        Вернуть {name: (next_date, age_turning)} для контактов,
        у которых ближайший ДР в интервале [0, days_ahead] от сегодня.
        """
        today = datetime.date.today()
        result: dict[str, Tuple[datetime.date, int]] = {}

        for rec in self.data.values():
            if not rec.birthday:
                continue
            month, day = rec.birthday.value.month, rec.birthday.value.day
            year = today.year
            # ближайшая дата ДР
            try:
                next_bd = datetime.date(year, month, day)
            except ValueError:  # 29 фев на невисокосный
                next_bd = datetime.date(year, 2, 28)
            if next_bd < today:  # уже прошёл – берём след. год
                try:
                    next_bd = datetime.date(year + 1, month, day)
                except ValueError:
                    next_bd = datetime.date(year + 1, 2, 28)
            delta = (next_bd - today).days
            if 0 <= delta <= days_ahead:
                result[rec.name.value] = (next_bd, next_bd.year - rec.birthday.value.year)

        return result


# ────────────────────────────────────────────────────────────────────────────
# Notes
# ────────────────────────────────────────────────────────────────────────────
class GeneralNote:
    def __init__(self, text: str, tags: List[str]):
        self.text = text.strip()
        self.tags = tags
        self.created_at = datetime.date.today()

    def __str__(self):
        tags = ", ".join(self.tags) if self.tags else "—"
        return f"{self.created_at.isoformat()}   [{tags}]   {self.text}"


class GeneralNoteBook:
    def __init__(self): self.notes: List[GeneralNote] = []

    def add_note(self, text: str, tags: List[str]): self.notes.append(GeneralNote(text, tags))

    def list_notes(self): return self.notes

    def search_by_tag(self, tag: str): return [n for n in self.notes if tag in n.tags]

# ────────────────────────────────────────────────────────────────────────────
# user‑scoped storage
# ────────────────────────────────────────────────────────────────────────────
def user_path(username: str, base: str) -> str:
    root = os.path.join("data", username.lower())
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, base)

# ────────────────────────────────────────────────────────────────────────────
# Persistence
# ────────────────────────────────────────────────────────────────────────────
DATA_FILE, NOTES_FILE = "addressbook.pkl", "notesbook.pkl"


def _save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def _load(path, factory):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.PickleError):
        return factory()

def load_data(username: str):
    return _load(user_path(username, DATA_FILE), AddressBook)

def load_notes(username: str):
    return _load(user_path(username, NOTES_FILE), GeneralNoteBook)

def save_data(username: str, ab):
    _save(ab, user_path(username, DATA_FILE))

def save_notes(username: str, nb):
    _save(nb, user_path(username, NOTES_FILE))


USERS_FILE = "users.pkl"


def load_users():
    try:
        with open(USERS_FILE, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.PickleError):
        return {}


def save_users(users):
    with open(USERS_FILE, "wb") as f:
        pickle.dump(users, f)
# ────────────────────────────────────────────────────────────────────────────
# simple keyword match
# ────────────────────────────────────────────────────────────────────────────
def simple_match(query: str, note: "GeneralNote") -> bool:
    q_words = {w.lower() for w in re.findall(r"\w+", query)}
    text    = note.text.lower()
    tags    = " ".join(note.tags).lower()
    return all(any(word in field for field in (text, tags)) for word in q_words)
# ────────────────────────────────────────────────────────────────────────────
# GPT semantic prompt
# ────────────────────────────────────────────────────────────────────────────
SEM_PROMPT = """You are a semantic search assistant.
Below is a numbered list of notes. Each note has the format
<index>: <text>  [tags: <tag1>, <tag2>, ...]

User will send a search query in Russian, Ukrainian or English.
Return ONLY the indices (space‑separated) of up to five notes
that are truly relevant. If nothing fits, return an empty string.
"""

# ────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────
def ok(msg): return f"[green]✔ {msg}[/]"


def prompt_validated(prompt: str, factory: Optional[Type[Field]] = None,
                     allow_blank=True) -> str:
    while True:
        raw = console.input(prompt).strip()
        if not raw and allow_blank:
            return ""
        if factory is None:
            return raw
        try:
            factory(raw)
            return raw
        except ValueError as e:
            console.print(f"[red]{e}[/]")


def _panel_body(rec: Record, extra=""):
    phones = ", ".join(p.value for p in rec.phones) or "—"
    bday = rec.birthday.value.strftime("%d.%m.%Y") if rec.birthday else "—"
    notes_list = getattr(rec, "contact_notes", [])
    notes = " | ".join(notes_list) if notes_list else "—"
    body = (
        f"[b]📞[/b] {phones}\n"
        f"[b]📧[/b] {rec.email.value or '—'}\n"
        f"[b]📍[/b] {rec.address.value or '—'}\n"
        f"[b]🎂[/b] {bday}\n"
        f"[b]📝[/b] {notes}"
    )
    return body + (f"\n{extra}" if extra else "")


def show_records(recs: List[Record]):
    if not recs:
        console.print("[dim]No contacts.[/]")
        return
    console.print(Columns(
        [Panel(_panel_body(r),
               title=f"{r.name.value.upper()} {r.surname.value.upper()}".strip(), border_style="cyan")
         for r in recs],
        equal=True, expand=True))


def show_birthdays(book: AddressBook, matches):
    if not matches:
        console.print("🎉 No birthdays in this period.")
        return

    ordered = sorted(matches.items(), key=lambda x: x[1][0])
    panels = []

    for key, (dt, age) in ordered:
        rec = book.find(key)                       
        full_name = f"{rec.name.value.title()} {rec.surname.value.title()}".strip()
        panels.append(
            Panel(
                _panel_body(rec,
                            extra=f"🎂 {dt.strftime('%d.%m.%Y')} / {age} y"),
                title=full_name,
                border_style="magenta"
            )
        )

    console.print(Columns(panels, equal=True, expand=True))



def help_msg(section="contacts"):
    mapping = CONTACT_DESC if section == "contacts" else NOTE_DESC
    # console.print()
    # for cmd, desc in mapping.items():
    #     console.print(f"[cyan bold]{cmd}[/]  {desc}")

    table = Table(title="\n📘 Команди для роботи з нотатками", header_style="bold blue", style="bold bright_cyan")

    table.add_column("Команда", justify="center", style="bold deep_sky_blue1", no_wrap=True)
    table.add_column("Опис", justify="center", style="white")

    for cmd, desc in mapping.items():
        table.add_row(f"[green]{cmd}[/green]", desc)
    console.print(table)


def input_error(fn):
    def wrap(parts, *ctx):
        try:
            return fn(parts, *ctx)
        except (KeyError, IndexError):
            return "[red]Invalid command or args.[/]"
        except ValueError as e:
            return f"[red]{e}[/]"

    return wrap

def register(users):
    print("===== New User Registration =====")
    while True:
        username = input("Enter your name >>> ").strip()
        if username in users:
            print(f"User {username} already registered.")
        else:
            break
    password = getpass.getpass("Enter a password >>> ").strip()
    users[username] = password
    save_users(users)
    return username
def login(users):
    print("===== Login =====")
    username = input("Login >>> ").strip()
    password = input("Password >>> ").strip()
    if users.get(username) == password:
        return username
    else:
        print(f"Invalid credentials. Check you login or password.")
        return None
# ────────────────────────────────────────────────────────────────────────────
# Argument spec
# ────────────────────────────────────────────────────────────────────────────
ARG_SPEC = {
    "change_": 1, "remove-phone": 2, "phone": 1, "delete": 1,
    "add-birthday": 2, "show-birthday": 1, "add-contact-note": 2,
    "change-address": 2, "change-email": 2, "search": 1, "birthdays": 1,
    # notes
    "add-tag": 2, "search-tag": 1, "search-note": 1,
}
CONTACT_CMDS = list(CONTACT_DESC.keys())
NOTE_CMDS = list(NOTE_DESC.keys())


def collect_args(cmd):
    prompts = {
        "remove-phone": ["Contact name: ", "Phone: "],
        "phone": ["Contact name: "],
        "delete": ["Contact name: "],
        "add-birthday": ["Contact name: ", "Birthday DD.MM.YYYY: "],
        "show-birthday": ["Name or surname: "],
        "add-contact-note": ["Contact name: ", "Note: "],
        "search": ["Query: "],
        "birthdays": ["Days from today (N): "],
        # notes
        "add-tag": ["Note index: ", "Tags (comma): "],
        "search-tag": ["Tag: "],
        "search-note": ["Phrase: "],
    }
    answers = [console.input(p).strip() for p in prompts.get(cmd, [])]
    if cmd == "add-tag" and len(answers) == 2:
        idx, tags = answers
        return [idx] + [t for t in re.split(r"[ ,]+", tags) if t]
    return answers


# ────────────────────────────────────────────────────────────────────────────
# Handlers
# ────────────────────────────────────────────────────────────────────────────
def make_key(name: str, surname: str = "") -> str:
    return f"{name} {surname}".strip().lower()


def make_key_from_input(fullname: str) -> str:
    parts = fullname.strip().split(maxsplit=1)
    return make_key(*parts)


def get_record_key(name: str, book: AddressBook) -> Optional[str]:
    name_parts = name.strip().split(maxsplit=1)
    if not name_parts:
        return None

    ##key = make_key_from_input(name)
    ##if key in book.data:
    ##    return key

    # Partial match fallback
    matches = [k for k in book.data if all(part.lower() in k for part in name_parts)]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        console.print("[yellow]Multiple matches found:[/]")
        for i, k in enumerate(matches, 1):
            console.print(f"{i}. {k.title()}")
        idx = console.input("Select number >>> ").strip()
        if idx.isdigit() and 1 <= int(idx) <= len(matches):
            return matches[int(idx) - 1]
    return None


@input_error
def handle_contact(parts, ab: AddressBook):
    cmd, *args = parts
    if cmd in ("hello", "help"):
        return help_msg("contacts")

    # add/update
    if cmd == "add":
        if not args:
            name = prompt_validated("Enter name: ", allow_blank=False)
            surname = prompt_validated("Enter surname: ")
            phone = prompt_validated("Enter phone (10 digits): ", Phone)
            email = prompt_validated("Enter email: ", Email)
            address = prompt_validated("Enter address: ")
        else:
            name, *rest = args
            surname = rest[0] if rest else ""
            phone = rest[1] if len(rest) > 1 else ""
            email = rest[2] if len(rest) > 2 else ""
            address = " ".join(rest[3:]) if len(rest) > 3 else ""
        key = make_key(name, surname)
        rec = ab.data.get(key)
        if rec:
            if phone: rec.add_phone(phone)
            if surname: rec.surname = Surname(surname)
            if email: rec.email = Email(email)
            if address: rec.address = Address(address)
            return ok("Contact updated.")
        rec = Record(name, surname, Address(address) if address else "", Email(email) if email else "")
        if phone: rec.add_phone(phone)
        ab.data[key] = rec
        return ok("Contact added.")

    # phone change
    if cmd == "change":
        name_input = input("Which contact do you want to change? >>> ").strip()
        normalized_name = get_record_key(name_input, ab)
        if not normalized_name:
            return "Ooops. Contact not found :-("

        record = ab.data[normalized_name]

        field = input("What do you want to change in this contact? (phone / email / address) >>> ").strip().lower()

        if field == "phone":
            new_phone = input("Enter new phone >>> ").strip()
            record.phones = []
            record.add_phone(new_phone)
            return f"Phone updated for {normalized_name.capitalize()}"

        elif field == "email":
            new_email = input("Enter new email >>> ").strip()
            record.update_email(new_email)
            return f"Email updated for {normalized_name.capitalize()}"

        elif field == "address":
            new_address = input("Enter new address >>> ").strip()
            record.update_address(new_address)
            return f"Address updated for {normalized_name.capitalize()}"

        else:
            return "Unknown command. Choose from: phone / email / address"

    if cmd == "remove-phone":
        name, phone = args
        ab.find(name).remove_phone(phone)
        return ok("Phone removed.")
    if cmd == "phone":
        name, = args
        return ", ".join(p.value for p in ab.find(name).phones) or "No phones."
    if cmd == "delete":
        name, = args
        ab.delete(name)
        return ok("Contact deleted.")

    # search/list
    if cmd == "all":
        show_records(list(ab.data.values()))
        return ""
    if cmd == "search":
        q, = args
        hits = [r for r in ab.data.values()
                if q.lower() in r.name.value.lower()
                or q.lower() in r.surname.value.lower()
                or any(q in p.value for p in r.phones)
                or any(q.lower() in note.lower() for note in r.contact_notes)]
        show_records(hits)
        return ""

    # birthdays
    if cmd == "add-birthday":
        name, date = args
        ab.find(name).add_birthday(date)
        return ok("Birthday added.")
    if cmd == "show-birthday":
        key, = args
        matches = [r for r in ab.data.values()
                   if key.lower() in (r.name.value.lower(), r.surname.value.lower())]
        if not matches:
            raise KeyError("Contact not found.")
        return "\n".join(f"{r.name.value} {r.surname.value}: "
                         f"{r.birthday.value.strftime('%d.%m.%Y') if r.birthday else '—'}"
                         for r in matches)
    if cmd == "birthdays":
        days, = args
        if not days.isdigit():
            raise ValueError("Enter non‑negative integer.")
        matches = ab.upcoming(int(days))
        show_birthdays(ab, matches)
        return ""

    # misc
    if cmd == "add-contact-note":
        name, *note = args
        ab.find(name).add_contact_note(" ".join(note))
        return ok("Note added.")

    if cmd in ("back", "exit", "close"):
        return "BACK"
    return "Unknown contact command."


@input_error
def handle_notes(parts, nb: GeneralNoteBook):
    cmd, *args = parts
    if cmd in ("hello", "help"):
        return help_msg("notes")
    if cmd == "add-note":
        text = " ".join(args) if args else console.input("Text: ")
        if not text.strip():
            raise ValueError("Empty note.")
        nb.add_note(text, [])
        if console.input("Add tags? (y/n): ").lower().startswith("y"):
            tags = re.split(r"[ ,]+", console.input("Tags: "))
            nb.notes[-1].tags.extend([t for t in tags if t])
        return ok("Note saved.")
    if cmd == "list-notes":
        notes = nb.list_notes()
        if not notes:
            console.print("[dim]No notes.[/]")
            return ""

        table = Table(show_header=True, header_style="bold blue",
                      box=None, expand=True)
        table.add_column("#", justify="right", style="bold cyan", no_wrap=True)
        table.add_column("Date", style="bright_cyan", no_wrap=True)
        table.add_column("Tags", style="green")
        table.add_column("Text", style="white")

        for i, n in enumerate(notes, 1):
            tags = ", ".join(n.tags) if n.tags else "—"
            table.add_row(str(i), n.created_at.isoformat(), tags, n.text)

        console.print(table)
        return ""

    if cmd == "add-tag":
        idx, *tags = args
        nb.notes[int(idx) - 1].tags.extend(tags)
        return ok("Tags added.")
    if cmd == "search-tag":
        tag = args[0] if args else console.input("Tag: ")
        res = nb.search_by_tag(tag)
        console.print("\n".join(str(n) for n in res) or f"No notes with tag '{tag}'.")
        return ""
    if cmd == "search-note":
        if not nb.notes:
            return "[dim]No notes to search.[/]"
        query = " ".join(args) if args else console.input("Query: ")

        # ---------- 1) быстрый keyword‑фильтр ----------
        hits = [i for i, n in enumerate(nb.notes)
                if simple_match(query, n)]
        if hits:
            console.print("[green]Keyword match:[/]")
            console.print("\n".join(f"{i + 1}. {nb.notes[i]}" for i in hits))
            return ""

        # ---------- 2) GPT‑семантика (если ключами не получилось) ----------
        if _client is None:
            return "[yellow]AI search disabled (no key.txt).[/]"

        catalog = "\n".join(
            f"{idx + 1}: {n.text}  [tags: {', '.join(n.tags) or '—'}]"
            for idx, n in enumerate(nb.notes)
        )
        sys_msg = SEM_PROMPT + "\n" + catalog
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            top_p=0.1,
            max_tokens=20,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": query}
            ]
        )
        idxs = [int(x) for x in re.findall(r"\d+", resp.choices[0].message.content)]
        if not idxs:
            console.print("[dim]No semantic matches.[/]")
            return ""
        console.print("[magenta]Semantic match:[/]")
        console.print("\n".join(f"{i}. {nb.notes[i - 1]}" for i in idxs))
        return ""


# ────────────────────────────────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────────────────────────────────
def main():
    users = load_users()
    print("Welcome to the assistant bot!")

    while True:
        choice = input("Do you want to (l)ogin or (r)egister? >>> ").strip()
        if choice == "r":
            username = register(users)
            break
        elif choice == "l":
            username = login(users)
            if username:
                break
        else:
            print("Incalid input. Please enter 'l' for login or 'r' for register." )

    print(f"Hello, {username.capitalize()}, glad to see you again!")
    ab = load_data(username)
    nb = load_notes(username)
    mode = "main"
    console.print("\nWellcome to [bold yellow]SYTObook[/] – your personal contacts and notes assistant 🤖\n")

    if _client is None:
        console.print("[yellow]AI functions disabled (no key.txt).[/]")

    while True:
        try:
            # main menu
            if mode == "main":
                choice = console.input(
                    "\n[bold]Choose a mode > [orchid]contacts[/] | [navajo_white1]notes[/] or exit:[/] ").strip().lower()
                if choice in ("exit", "close"):
                    save_data(username, ab)
                    save_notes(username, nb)
                    console.print(ok("Data saved. Bye!"));
                    break
                if choice in ("contacts", "notes"):
                    mode = choice
                    help_msg(mode)
                    continue
                console.print("Unknown mode.")
                continue

            # contacts
            if mode == "contacts":
                raw = console.input("\n[bold italic][orchid]Contacts[/]>>> Command :[/]").strip()
                if raw in ("exit", "close"):
                    save_data(username, ab)
                    save_notes(username, nb)
                    console.print(ok("Data saved. Bye!"));
                    break
                if raw == "back": mode = "main"; continue
                parts = raw.split()
                if not parts: continue
                cmd = parts[0]
                if cmd not in CONTACT_CMDS:
                    sug = suggest_correction(raw, CONTACT_DESC)
                    if sug and console.input(f"Did you mean '{sug}'? (y/n): ").lower().startswith("y"):
                        parts = [sug] + collect_args(sug)
                    else:
                        console.print("Unknown command.");
                        continue
                need = ARG_SPEC.get(parts[0], 0)
                if len(parts) - 1 < need: parts += collect_args(parts[0])
                res = handle_contact(parts, ab)
                if res == "BACK":
                    mode = "main"
                elif res:
                    console.print(res)

            # notes
            if mode == "notes":
                raw = console.input("\n[italic][navajo_white1]Notes[/]>>> Command :[/]").strip()
                if raw in ("exit", "close"):
                    save_data(username, ab)
                    save_notes(username, nb)
                    console.print(ok("Data saved. Bye!"));
                    break
                if raw == "back": mode = "main"; continue
                parts = raw.split()
                if not parts: continue
                cmd = parts[0]
                if cmd not in NOTE_CMDS:
                    sug = suggest_correction(raw, NOTE_DESC)
                    if sug and console.input(f"Did you mean '{sug}'? (y/n): ").lower().startswith("y"):
                        parts = [sug] + collect_args(sug)
                    else:
                        console.print("Unknown command.");
                        continue
                need = ARG_SPEC.get(parts[0], 0)
                if len(parts) - 1 < need: parts += collect_args(parts[0])
                res = handle_notes(parts, nb)
                if res == "BACK":
                    mode = "main"
                elif res:
                    console.print(res)

        except KeyboardInterrupt:
            console.print("\nInterrupted. Saving …")
            save_data(username, ab)
            save_notes(username, nb)
            break


if __name__ == "__main__":
    main()