from __future__ import annotations

import unittest

from src.ingestion.ingest_inmet import select_latest_station_members


class SelectLatestStationMembersTest(unittest.TestCase):
    def test_keeps_only_latest_cumulative_file_for_each_station(self) -> None:
        recife_august = (
            "INMET_NE_PE_A301_RECIFE_01-01-2026_A_31-08-2026.CSV"
        )
        recife_september = (
            "INMET_NE_PE_A301_RECIFE_01-01-2026_A_30-09-2026.CSV"
        )
        maceio_september = (
            "INMET_NE_AL_A303_MACEIO_01-01-2026_A_30-09-2026.CSV"
        )

        selected, superseded = select_latest_station_members(
            [recife_august, maceio_september, recife_september]
        )

        self.assertEqual(
            selected,
            sorted([recife_september, maceio_september]),
        )
        self.assertEqual(superseded, [recife_august])

    def test_preserves_member_when_filename_cannot_be_classified(self) -> None:
        unusual_name = "dados_estacao_sem_periodo.csv"

        selected, superseded = select_latest_station_members([unusual_name])

        self.assertEqual(selected, [unusual_name])
        self.assertEqual(superseded, [])


if __name__ == "__main__":
    unittest.main()
