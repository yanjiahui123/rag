# Clean Code Rules

This project uses these rules for new code and for files touched during feature work.

## Data Models

- Use `pydantic.BaseModel` for structured models.
- Do not add new `@dataclass` models.
- Keep model conversion methods explicit, for example `to_metadata()` or `to_parsed_block()`.

## Function Size

- Keep each new or modified function at 50 lines or fewer.
- If a function grows past 50 lines, split it by responsibility before adding more behavior.

## Compatibility

- Preserve existing public entry points when refactoring shared pipeline code.
- Add tests before changing behavior.
- Keep metadata payloads explicit and serializable.
