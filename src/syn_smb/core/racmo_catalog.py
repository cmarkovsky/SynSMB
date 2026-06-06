"""
racmo_catalog.py
================
Registry for RACMO SMB data files, organised by a three-level hierarchy:

  region    — broadest (e.g. 'West_Antarctica', 'East_Antarctica')
  subregion — intermediate IMBIE/MEaSUREs code (e.g. 'Ipp-J', 'Ap-B')
  name      — unique glacier/basin identifier (e.g. 'PineIsland')

The registry key is the unique NAME. region and subregion are metadata
stored on BasinEntry and used for filtering via paths().

Typical usage
-------------
    from racmo_catalog import RACMOCatalog

    catalog = RACMOCatalog("./data/sectors_smb")
    catalog.summarize()

    # All basins
    results = multi_basin_run(catalog.paths())

    # Filter by level
    results = multi_basin_run(catalog.paths(region="West"))
    results = multi_basin_run(catalog.paths(subregion="Ipp-J"))
    results = multi_basin_run(catalog.paths(names=["PineIsland", "Thwaites"]))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import math

from .data_loader import SMBDataLoader


# ======================================================================
# Utilities
# ======================================================================

def _clean(value) -> str | None:
    """Convert NaN / None / empty string to None; strip valid strings."""
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return s if s else None


# ======================================================================
# Known WAIS basins — pre-seeded with all three levels
# ======================================================================

KNOWN_BASINS: dict[str, dict] = {
    "PineIsland": {"region": "West_Antarctica", "subregion": "Ipp-J",  "name": "PineIsland"},
    "Thwaites":   {"region": "West_Antarctica", "subregion": "Ipp-H",  "name": "Thwaites"},
    "Getz":       {"region": "West_Antarctica", "subregion": "Ipp-F",  "name": "Getz"},
    "Dotson":     {"region": "West_Antarctica", "subregion": "Ipp-G",  "name": "Dotson"},
    "Crosson":    {"region": "West_Antarctica", "subregion": "Ipp-G",  "name": "Crosson"},
    "Abbott":     {"region": "West_Antarctica", "subregion": "Ipp-E",  "name": "Abbott"},
    "Kohler":     {"region": "West_Antarctica", "subregion": "Ipp-G",  "name": "Kohler"},
}

# Approximate SMB mean range (m w.e. a⁻¹) per NAME for sanity checks
_EXPECTED_MEAN: dict[str, tuple[float, float]] = {
    "PineIsland": (0.01, 0.15),
    "Thwaites":   (0.01, 0.20),
    "Getz":       (0.30, 0.80),
    "Dotson":     (0.20, 0.60),
    "Crosson":    (0.20, 0.60),
    "Abbott":     (0.10, 0.50),
    "Kohler":     (0.01, 0.15),
}


# ======================================================================
# BasinEntry
# ======================================================================

@dataclass
class BasinEntry:
    """
    Three-level metadata for a single registered basin.

    Hierarchy (most specific → broadest):
      name      : str | None  — unique glacier id (e.g. 'PineIsland')
      subregion : str | None  — IMBIE/MEaSUREs code (e.g. 'Ipp-J')
      region    : str | None  — broad area (e.g. 'West_Antarctica')
    """
    path:      Path
    region:    str | None = None
    subregion: str | None = None
    name:      str | None = None

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def display_name(self) -> str:
        parts = [p for p in (self.region, self.subregion, self.name) if p]
        return " / ".join(parts) if parts else self.path.stem

    def __repr__(self) -> str:
        return (
            f"BasinEntry("
            f"name={self.name!r}, "
            f"subregion={self.subregion!r}, "
            f"region={self.region!r}, "
            f"exists={self.exists})"
        )


# ======================================================================
# RACMOCatalog
# ======================================================================

@dataclass
class RACMOCatalog:
    """
    Registry of RACMO SMB files with three-level hierarchy.

    Parameters
    ----------
    data_dir : str or Path
    var : str
        NetCDF variable name. Default 'smbgl'.
    file_pattern : str
        Auto-discovery pattern. Default '{basin}_smb.nc'.

    Examples
    --------
    >>> cat = RACMOCatalog("./data/sectors_smb")
    >>> cat.summarize()
    >>> cat.paths(region="West")
    >>> cat.paths(subregion="Ipp-J")
    >>> cat.by_region()
    >>> cat.by_subregion()
    """

    data_dir:     str | Path = "."
    var:          str        = "smbgl"
    file_pattern: str        = "{basin}_smb.nc"

    _registry: dict[str, BasinEntry] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self._auto_discover()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _auto_discover(self) -> None:
        for key, meta in KNOWN_BASINS.items():
            p = self.data_dir / self.file_pattern.format(basin=key)
            if p.exists():
                self._registry[key] = BasinEntry(
                    path=p,
                    region=meta["region"],
                    subregion=meta["subregion"],
                    name=meta["name"],
                )

    def _resolve(self, key: str) -> BasinEntry:
        if key not in self._registry:
            raise KeyError(
                f"'{key}' not registered. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[key]

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        key:       str,
        path:      str | Path,
        region:    str | None = None,
        subregion: str | None = None,
        name:      str | None = None,
    ) -> RACMOCatalog:
        """
        Register one basin with its three-level metadata.

        Parameters
        ----------
        key       : registry key and filename stem (the unique NAME).
        path      : path to NetCDF file.
        region    : broad region (e.g. 'West_Antarctica').
        subregion : intermediate code (e.g. 'Ipp-J').
        name      : unique glacier name — defaults to key if not given.
        """
        p = Path(path)
        if not p.is_absolute():
            p = self.data_dir / p

        known = KNOWN_BASINS.get(key, {})
        self._registry[key] = BasinEntry(
            path      = p,
            region    = _clean(region)    or _clean(known.get("region")),
            subregion = _clean(subregion) or _clean(known.get("subregion")),
            name      = _clean(name)      or _clean(known.get("name")) or key,
        )
        return self

    def register_many(
        self,
        entries: dict[str, str | Path | dict],
        region: str | None = None,
    ) -> RACMOCatalog:
        """
        Register multiple basins.

        entries can be {key: path} or
        {key: {'path':..., 'region':..., 'subregion':..., 'name':...}}.
        region is a default applied to simple-path entries.
        """
        for key, value in entries.items():
            if isinstance(value, dict):
                self.register(
                    key,
                    value["path"],
                    region=value.get("region", region),
                    subregion=value.get("subregion"),
                    name=value.get("name"),
                )
            else:
                self.register(key, value, region=region)
        return self

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def path(self, key: str) -> Path:
        return self._resolve(key).path

    def entry(self, key: str) -> BasinEntry:
        return self._resolve(key)

    def paths(
        self,
        keys:       list[str] | None = None,
        region:     str | None = None,
        subregion:  str | None = None,
        names:      list[str] | None = None,
        require_exists: bool = True,
    ) -> dict[str, str]:
        """
        Return {key: path_str} for use with multi_basin_run().

        Filtering (applied in order, all are AND conditions):
          keys      : explicit list of registry keys
          region    : case-insensitive partial match on entry.region
          subregion : case-insensitive partial match on entry.subregion
          names     : list of entry.name values to include
        """
        candidates = keys if keys is not None else list(self._registry.keys())

        if region is not None:
            r = region.lower()
            candidates = [k for k in candidates
                          if self._registry[k].region is not None
                          and r in self._registry[k].region.lower()]

        if subregion is not None:
            s = subregion.lower()
            candidates = [k for k in candidates
                          if self._registry[k].subregion is not None
                          and s in self._registry[k].subregion.lower()]

        if names is not None:
            name_set = {n.lower() for n in names}
            candidates = [k for k in candidates
                          if self._registry[k].name is not None
                          and self._registry[k].name.lower() in name_set]

        result = {}
        for key in candidates:
            e = self._resolve(key)
            if not e.path.exists():
                if require_exists:
                    raise FileNotFoundError(
                        f"File not found for '{key}': {e.path}"
                    )
                continue
            result[key] = str(e.path)
        return result

    def available(self) -> list[str]:
        return [k for k, e in self._registry.items() if e.exists]

    def missing(self) -> list[str]:
        return [k for k, e in self._registry.items() if not e.exists]

    def regions(self) -> list[str]:
        return sorted({
            e.region for e in self._registry.values()
            if isinstance(e.region, str)
        })

    def subregions(self, region: str | None = None) -> list[str]:
        entries = self._registry.values()
        if region is not None:
            r = region.lower()
            entries = [e for e in entries
                       if e.region is not None and r in e.region.lower()]
        return sorted({e.subregion for e in entries if isinstance(e.subregion, str)})

    def by_region(self) -> dict[str, list[str]]:
        """Return {region: [key, ...]} grouped by broad region."""
        grouped: dict[str, list[str]] = defaultdict(list)
        for key, e in self._registry.items():
            grouped[e.region or "Unknown"].append(key)
        return dict(sorted(grouped.items()))

    def by_subregion(self, region: str | None = None) -> dict[str, list[str]]:
        """
        Return {subregion: [key, ...]} grouped by intermediate subregion code.
        Optionally filtered to a single broad region.
        """
        grouped: dict[str, list[str]] = defaultdict(list)
        for key, e in self._registry.items():
            if region is not None:
                if e.region is None or region.lower() not in e.region.lower():
                    continue
            grouped[e.subregion or "Unknown"].append(key)
        return dict(sorted(grouped.items()))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, verbose: bool = True) -> dict[str, dict]:
        report = {}
        for key, entry in self._registry.items():
            rec = {
                "region":    entry.region,
                "subregion": entry.subregion,
                "name":      entry.name,
                "exists":    entry.exists,
                "loadable":  False,
                "n_obs":     None,
                "mean":      None,
                "mean_ok":   None,
                "error":     None,
            }
            if rec["exists"]:
                try:
                    smb = SMBDataLoader(str(entry.path), var=self.var).load()
                    rec["loadable"] = True
                    rec["n_obs"]    = smb.sizes["time"]
                    rec["mean"]     = float(smb.mean())
                    check_key = entry.name or key
                    if check_key in _EXPECTED_MEAN:
                        lo, hi = _EXPECTED_MEAN[check_key]
                        rec["mean_ok"] = lo <= rec["mean"] <= hi
                    else:
                        rec["mean_ok"] = True
                except Exception as exc:
                    rec["error"] = str(exc)
            report[key] = rec

        if verbose:
            self._print_validate(report)
        return report

    def _print_validate(self, report: dict) -> None:
        print("RACMOCatalog validation")
        print(f"  data_dir : {self.data_dir}")
        print(f"  var      : {self.var}")
        print()
        kw = max((len(k) for k in report), default=4) + 2
        nw = max((len(v["name"] or "") for v in report.values()), default=4) + 2
        sw = max((len(v["subregion"] or "") for v in report.values()), default=8) + 2
        rw = max((len(v["region"] or "") for v in report.values()), default=6) + 2
        header = (f"  {'Key':<{kw}} {'Name':<{nw}} {'Subregion':<{sw}}"
                  f" {'Region':<{rw}} {'Ex':>3} {'Ld':>3}"
                  f" {'n_obs':>6} {'mean':>8} {'OK':>3}")
        print(header)
        print("  " + "─" * (len(header) - 2))
        for key, r in report.items():
            ex  = "✓" if r["exists"]   else "✗"
            ld  = "✓" if r["loadable"] else ("─" if not r["exists"] else "✗")
            n   = str(r["n_obs"])     if r["n_obs"] is not None  else "─"
            m   = f"{r['mean']:.4f}"  if r["mean"]  is not None  else "─"
            ok  = ("✓" if r["mean_ok"] else "⚠") if r["mean_ok"] is not None else "─"
            err = f"  ← {r['error']}" if r["error"] else ""
            print(f"  {key:<{kw}} {(r['name'] or ''):<{nw}}"
                  f" {(r['subregion'] or ''):<{sw}}"
                  f" {(r['region'] or ''):<{rw}}"
                  f" {ex:>3} {ld:>3} {n:>6} {m:>8} {ok:>3}{err}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarize(self) -> None:
        """Print a three-level grouped summary."""
        print("RACMOCatalog")
        print(f"  data_dir : {self.data_dir}")
        print(f"  var      : {self.var}")
        print(f"  Basins   : {len(self._registry)}"
              f"  (available: {len(self.available())},"
              f"  missing: {len(self.missing())})")
        print()

        # Group: region → subregion → names
        tree: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for key, e in self._registry.items():
            reg = e.region    or "Unknown_region"
            sub = e.subregion or "Unknown_subregion"
            tree[reg][sub].append(key)

        for reg in sorted(tree):
            print(f"  [{reg}]")
            for sub in sorted(tree[reg]):
                keys = sorted(tree[reg][sub])
                statuses = [
                    "✓" if self._registry[k].exists else "✗"
                    for k in keys
                ]
                names = [
                    self._registry[k].name or k for k in keys
                ]
                line = "  ".join(f"{s} {n}" for s, n in zip(statuses, names))
                print(f"    ({sub})  {line}")
            print()

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        data_dir: str | Path,
        var: str = "smbgl",
        file_pattern: str = "{basin}_smb.nc",
        extra: dict | None = None,
    ) -> RACMOCatalog:
        cat = cls(data_dir=data_dir, var=var, file_pattern=file_pattern)
        if extra:
            cat.register_many(extra)
        return cat

    @classmethod
    def from_dict(
        cls,
        entries: dict[str, str | Path | dict],
        var: str = "smbgl",
        default_region: str | None = None,
    ) -> RACMOCatalog:
        """
        Create from explicit mapping.

        entries: {key: path}  or
                 {key: {'path':..., 'region':..., 'subregion':..., 'name':...}}
        """
        first = next(iter(entries.values()))
        p0 = Path(first["path"] if isinstance(first, dict) else first)
        data_dir = p0.parent if p0.is_absolute() else Path(".")
        cat = cls(data_dir=data_dir, var=var)
        cat._registry.clear()
        cat.register_many(entries, region=default_region)
        return cat

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RACMOCatalog(registered={len(self._registry)}, "
            f"available={len(self.available())}, "
            f"regions={self.regions()})"
        )

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __iter__(self):
        return iter(self._registry)