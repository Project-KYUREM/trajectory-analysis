"""Machinery shared by the mission ``Config`` dataclasses.

Each mission's ``config.py`` declares its own fields in plain sight and
subclasses ``ConfigBase``, which supplies validation, the opt-flag
resolution, the margins-and-assumptions registry, the JSON record and the
echo. A subclass sets or overrides:

    CHOICES           {field: allowed values}, checked by ``validate``.
    ASSUMPTIONS       ``Assumption`` entries, one per field of the config's
                      "margins and assumptions" group.
    auto_opt_flags()  the automatic (opt_direction, opt_t_peri) rule.
    check()           numeric checks beyond CHOICES (raise ValueError).
    warnings()        input sanity warnings, printed with the echo and saved.
    derived()         derived values added to the record.

Field tags used in the CONFIG cells: [physics] changes the problem,
[numerics] changes how it is solved, [guess] only shapes the starting point,
[margin] is a margin, assumed value, weight or check tolerance listed in
``ASSUMPTIONS``.
"""
import functools
from dataclasses import asdict, dataclass, replace

ASSUMPTION_KINDS = {
    "margin": "safety or design margin on a limit",
    "requirement": "mission requirement chosen for the study",
    "model": "assumed model value or validity bound",
    "weight": "objective regularisation weight",
    "check": "verification tolerance (does not change the solution)",
}


@dataclass(frozen=True)
class Assumption:
    """One entry of a config's margins-and-assumptions group.

    Attributes:
        field (str): Config field holding the value.
        unit (str): Unit of the value ("" when dimensionless).
        kind (str): Key of ``ASSUMPTION_KINDS``.
        why (str): Why it has this value and what it costs, in one line.
    """
    field: str
    unit: str
    kind: str
    why: str


def _lists(o):
    """Return ``o`` with every tuple, at any depth, turned into a list."""
    if isinstance(o, dict):
        return {k: _lists(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_lists(v) for v in o]
    return o


def format_value(v, unit=""):
    """Return a value with its unit as short text (floats in ``%g``)."""
    text = f"{v:g}" if isinstance(v, float) else str(v)
    return f"{text} {unit}".rstrip() if unit and v is not None else text


class ConfigBase:
    """Validation, opt-flag resolution, assumptions, record and echo of a mission config."""

    CHOICES = {}
    ASSUMPTIONS = ()

    # ------------------------------------------------------------ hooks
    def auto_opt_flags(self):
        """Return the automatic ``(opt_direction, opt_t_peri)`` of this config."""
        raise NotImplementedError

    def check(self):
        """Run the mission's numeric checks (raise ValueError on a bad input)."""

    def warnings(self):
        """Return the input sanity warnings (printed with the echo and saved)."""
        return []

    def derived(self):
        """Return the derived values added to the record."""
        return {}

    # ------------------------------------------------------------ shared
    def resolved(self):
        """Return a copy with ``opt_direction`` and ``opt_t_peri`` set to booleans.

        A None flag takes the value of ``auto_opt_flags``; an explicit
        boolean is kept.

        Returns:
            ConfigBase: Copy with both flags resolved.
        """
        auto_dir, auto_t = self.auto_opt_flags()
        return replace(
            self,
            opt_direction=auto_dir if self.opt_direction is None else self.opt_direction,
            opt_t_peri=auto_t if self.opt_t_peri is None else self.opt_t_peri)

    def validate(self):
        """Check every categorical input against ``CHOICES``, then run ``check``.

        Raises:
            ValueError: If an option string is unknown or a numeric check fails.
        """
        for name, allowed in self.CHOICES.items():
            value = functools.reduce(getattr, name.split("."), self)
            if value not in allowed:
                raise ValueError(f"{name} must be one of {tuple(allowed)}, not {value!r}")
        self.check()

    def assumptions(self):
        """Return the margins and assumptions in effect, one dict per ``ASSUMPTIONS`` entry.

        Returns:
            list: Dicts with ``field``, ``value``, ``unit``, ``kind`` and ``why``.
        """
        return [dict(field=a.field, value=_lists(getattr(self, a.field)), unit=a.unit,
                     kind=a.kind, why=a.why) for a in self.ASSUMPTIONS]

    def record(self):
        """Return a JSON-ready dict of every input (opt flags resolved) and the derived values."""
        d = _lists(asdict(self.resolved()))
        d.update(_lists(self.derived()))
        return d

    def echo(self, stream=print):
        """Print every input and derived value ([margin] fields marked) and the input warnings."""
        stream("Config ([margin]: margins and assumptions group):")
        margins = {a.field for a in self.ASSUMPTIONS}
        for k, v in self.record().items():
            if isinstance(v, dict):
                stream(f"  {k}:")
                for vk, vv in v.items():
                    stream(f"    {vk:26s} = {vv}")
            else:
                stream(f"  {k:28s} = {v}{'   [margin]' if k in margins else ''}")
        for w in self.warnings():
            stream(f"  WARNING: {w}")

