"""TASK-719 unified entity-presentation resolver tests."""

from types import SimpleNamespace

from llm_bawt.entity_presentation import EntityPresentationResolver
from llm_bawt.message_authorship import AuthorReference


class Result:
    def __init__(self, rows):
        self.rows = rows

    def mappings(self):
        return self

    def all(self):
        return self.rows


class Connection:
    def __init__(self, engine):
        self.engine = engine

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, params):
        sql = str(statement)
        self.engine.calls.append((sql, params))
        if "FROM entity_profiles" in sql:
            return Result(self.engine.profile_rows)
        if "FROM bot_profiles" in sql:
            return Result(self.engine.bot_rows)
        raise AssertionError(sql)


class Engine:
    def __init__(self):
        self.calls = []
        self.profile_rows = [
            {
                "entity_type": "USER",
                "entity_id": "operator",
                "display_name": "Operator",
                "color": "emerald",
                "avatar": "🐕",
                "avatar_render": None,
            },
            {
                "entity_type": "bot",
                "entity_id": "snark",
                "display_name": None,
                "color": None,
                "avatar": None,
                "avatar_render": None,
            },
        ]
        self.bot_rows = [
            {
                "slug": "snark",
                "name": "Snark",
                "color": "cyan",
                "avatar": "🔧",
                "avatar_render": "data:image/svg+xml,bot",
            }
        ]

    def connect(self):
        return Connection(self)


def test_resolver_batches_profiles_and_bots_once() -> None:
    engine = Engine()
    resolver = EntityPresentationResolver(engine)
    authors = [
        AuthorReference.user("operator"),
        AuthorReference.user("operator"),
        AuthorReference.bot("snark"),
    ]

    result = resolver.resolve_many(authors)

    assert len(engine.calls) == 2
    assert result[("user", "operator")].to_dict() == {
        "entity_type": "user",
        "entity_id": "operator",
        "display_name": "Operator",
        "color": "emerald",
        "avatar": "🐕",
        "avatar_render": None,
        "status": "canonical",
    }
    assert result[("bot", "snark")].display_name == "Snark"
    assert result[("bot", "snark")].avatar_render == "data:image/svg+xml,bot"


def test_registered_bot_uses_registry_even_when_presentation_was_cleared() -> None:
    engine = Engine()
    legacy_bot = engine.profile_rows[1]
    legacy_bot.update(display_name="Old name", color="#ffffff", avatar="A", avatar_render="old-render")
    engine.bot_rows[0].update(name="New name", color=None, avatar=None, avatar_render=None)

    result = EntityPresentationResolver(engine).resolve_many([AuthorReference.bot("snark")])
    author = result[("bot", "snark")]

    assert author.display_name == "New name"
    assert author.color != legacy_bot["color"]
    assert author.avatar is None
    assert author.avatar_render is None


def test_bot_without_registry_row_keeps_legacy_presentation() -> None:
    engine = Engine()
    engine.bot_rows = []
    engine.profile_rows[1].update(display_name="Legacy bot", color="rose", avatar="A")
    result = EntityPresentationResolver(engine).resolve_many([AuthorReference.bot("snark")])
    author = result[("bot", "snark")]
    assert (author.display_name, author.color, author.avatar) == ("Legacy bot", "rose", "A")


def test_missing_entity_preserves_identity_but_becomes_unknown() -> None:
    engine = Engine()
    engine.profile_rows = []
    engine.bot_rows = []
    result = EntityPresentationResolver(engine).resolve_many(
        [AuthorReference.bot("deleted-bot")]
    )[("bot", "deleted-bot")]

    assert result.entity_type == "bot"
    assert result.entity_id == "deleted-bot"
    assert result.display_name == "Unknown bot"
    assert result.status == "unknown"
    assert result.color


def test_unresolved_legacy_author_uses_neutral_unknown() -> None:
    engine = Engine()
    result = EntityPresentationResolver(engine).resolve_many(
        [AuthorReference.unknown()]
    )[(None, None)]

    assert result.display_name == "Unknown participant"
    assert result.color == "slate"
    assert result.status == "unknown"
