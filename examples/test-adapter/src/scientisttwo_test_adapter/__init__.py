"""Separately installed trusted adapter used to verify entry-point discovery."""

from scientisttwo.live.adapters import CSVDevelopment


class InstalledCSV(CSVDevelopment):
    def identity(self):
        return {**super().identity(), "example_adapter": "installed-csv-v1"}
