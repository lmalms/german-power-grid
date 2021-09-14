from typing import Any, Dict, Union, Optional

import requests
from requests.exceptions import HTTPError
from bs4 import BeautifulSoup
import regex as re
import datetime as dt
import pytz
import numpy as np
import pandas as pd

from gpg.data_retrieval.config import *

# TODO: docstrings + readme for data retrieval package


class EntsoeClient:

    def __init__(
            self,
            api_token: str,
            default_document: Optional[Union[str, Document]] = None,
            default_process: Optional[Union[str, Process]] = None,
            default_area: Optional[Union[str, Area]] = None
    ):
        self.api_token = api_token
        self.default_document = default_document
        self.default_process = default_process
        self.default_area = default_area
        self._base_url = BASE_URL

    @property
    def default_document(self) -> Document:
        return self._default_document

    @property
    def default_process(self) -> Process:
        return self._default_process

    @property
    def default_area(self) -> Area:
        return self._default_area

    @default_document.setter
    def default_document(self, default_document: Union[str, Document]) -> None:
        if isinstance(default_document, str):
            self._default_document = self._infer_document_type(default_document)
        elif isinstance(default_document, Document):
            self._default_document = default_document
        else:
            self._default_document = None

    @default_process.setter
    def default_process(self, default_process: Union[str, Process]) -> None:
        if isinstance(default_process, str):
            self._default_process = self._infer_process_type(default_process)
        elif isinstance(default_process, Process):
            self._default_process = default_process
        else:
            self._default_process = None

    @default_area.setter
    def default_area(self, default_area: Union[str, Area]) -> None:
        if isinstance(default_area, str):
            self._default_area = self._infer_area(default_area)
        elif isinstance(default_area, Area):
            self._default_area = default_area
        else:
            self._default_area = None

    def __call__(self, start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
        """
        Same as running the default query over the specified time range. Returns the result of the query as a
        pd.DataFrame.
        Args:
        -----
            start_date: dt.datetime
                The start date for the query.
            end_date: dt.datetime
                The end date for the query.

        Returns:
        --------
            The data of the query (if successful) as a pd.DataFrame.
        """

        return self.default_query(start_date=start_date, end_date=end_date)

    def default_query(
            self,
            start_date: dt.datetime,
            end_date: dt.datetime,
            convert_utc_to_local_tz: bool = False
    ) -> pd.DataFrame:
        """
        Runs the default query and returns data as a pd.DataFrame

        Args:
        -----
            start_date: dt.datetime
                Start date for the default query.
            end_date: dt.datetime
                End date for the default query.

        Returns:
        --------
            pd.DataFrame
                The data of the query (if successful) as a DataFrame.
        """
        # Check that default attributes are defined.
        assert all([
            param is not None for param in [self._default_document, self.default_process, self.default_area]
        ]), "For default queries all default attributes (document, process and area) need to be defined. " \
            "At least one attribute is None."

        # Run query and extract data
        return self.extract_data(
            response=self.query(
                document=self.default_document,
                process=self.default_process.process_code,
                area=self.default_area.area_code,
                start_date=start_date,
                end_date=end_date
            ),
            convert_utc_to_local_tz=convert_utc_to_local_tz
        )

    def query(
            self,
            document: Union[str, Document],
            process: Union[str, Process],
            area: Union[str, Area],
            start_date: dt.datetime,
            end_date: dt.datetime,
            timeout: Optional[float] = 10.
    ) -> requests.Response:

        # Validate query
        self._validate_query(start_date=start_date, end_date=end_date)

        # Infer document type, if str given
        if isinstance(document, str):
            document = self._infer_document_type(document)

        # Infer process type, if str given
        if isinstance(process, str):
            process = self._infer_process_type(process)

        # Infer area code, if str given
        if isinstance(area, str):
            area = self._infer_area(area)

        # Make query
        response = requests.get(
            url=self._base_url,
            params=dict(
                securityToken=self.api_token,
                documentType=document.document_code,
                processType=process.process_code,
                outBiddingZone_Domain=area.area_code,
                periodStart=self._format_timestamp(timestamp=start_date, convert_local_tz_to_utc=True,
                                                   convert_utc_to_local_tz=False, local_tz=area.time_zone),
                periodEnd=self._format_timestamp(timestamp=end_date, convert_local_tz_to_utc=True,
                                                 convert_utc_to_local_tz=False, local_tz=area.time_zone)
            ),
            timeout=timeout
        )

        try:
            response.raise_for_status()
        except HTTPError as e:
            status_code = response.status_code
            message = e.args[0]
            reason = BeautifulSoup(response.text, "xml").find("text").get_text()
            raise EntsoeClientError(status_code=status_code, message=message, reason=reason) from None
        else:
            return response

    def extract_data(self, response: requests.Response, convert_utc_to_local_tz: bool = False) -> pd.DataFrame:
        soup = BeautifulSoup(response.text, "xml")
        local_tz = self.extract_query_details(response=response)["area"].time_zone
        dfs = []
        for ts in soup.find_all("TimeSeries"):
            start = self._format_timestamp(
                ts.start.get_text(),
                convert_utc_to_local_tz=convert_utc_to_local_tz,
                convert_local_tz_to_utc=False,
                local_tz=local_tz
            )
            delta = pd.to_timedelta(ts.resolution.get_text()).to_pytimedelta()

            dfs.append(
                pd.DataFrame(
                    data={
                        "timestamp": np.array([start + i * delta for i, _ in enumerate(ts.find_all("quantity"))]),
                        "quantity": np.array([float(quantity.string) for quantity in ts.find_all("quantity")])
                    }
                )
            )

        return pd.concat(dfs).reset_index(drop=True)

    def extract_query_details(
            self,
            response: requests.Response,
            convert_utc_to_local_tz: bool = False
    ) -> Dict[str, Any]:
        soup = BeautifulSoup(response.text, "xml")
        local_tz = self._infer_area(soup.find("outBiddingZone_Domain.mRID").get_text()).time_zone
        return dict(
            document=self._infer_document_type(soup.find("type").get_text()),
            process=self._infer_process_type(soup.find("process.processType").get_text()),
            area=self._infer_area(soup.find("outBiddingZone_Domain.mRID").get_text()),
            start_date=self._format_timestamp(
                soup.find("time_Period.timeInterval").start.get_text(),
                convert_utc_to_local_tz=convert_utc_to_local_tz,
                convert_local_tz_to_utc=False,
                local_tz=local_tz
            ),
            end_date=self._format_timestamp(
                soup.find("time_Period.timeInterval").end.get_text(),
                convert_utc_to_local_tz=convert_utc_to_local_tz,
                convert_local_tz_to_utc=False,
                local_tz=local_tz
            ),
            query_timestamp=self._format_timestamp(
                soup.find("createdDateTime").get_text(),
                convert_utc_to_local_tz=convert_utc_to_local_tz,
                convert_local_tz_to_utc=False,
                local_tz=local_tz
            )
        )

    @staticmethod
    def _validate_query(start_date: dt.datetime, end_date: dt.datetime) -> None:

        # TODO

        # validate dates
        assert start_date.tzinfo is None, "start date should be time zone naive."
        assert end_date.tzinfo is None, "end date should be time zone naive."

    @staticmethod
    def _infer_document_type(document: str) -> Document:
        if document in [name for name, _ in Document.__members__.items()]:
            return Document[document]
        elif document in [member.value for _, member in Document.__members__.items()]:
            return [member for _, member in Document.__members__.items() if member.value == document][0]

    @staticmethod
    def _infer_process_type(process: str) -> Process:
        if process in [name for name, _ in Process.__members__.items()]:
            return Process[process]
        elif process in [member.value for _, member in Process.__members__.items()]:
            return [member for _, member in Process.__members__.items() if member.value == process][0]

    @staticmethod
    def _infer_area(area: str) -> Area:
        if area in [name for name, _ in Area.__members__.items()]:
            return Area[area]
        elif area in [member.value[0] for _, member in Area.__members__.items()]:
            return [member for _, member in Area.__members__.items() if member.value[0] == area][0]

    @staticmethod
    def _format_timestamp(
            timestamp: Union[dt.datetime, str],
            convert_local_tz_to_utc: bool,
            convert_utc_to_local_tz: bool,
            local_tz: str
    ) -> Union[str, dt.datetime]:

        local_tz = pytz.timezone(local_tz)

        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp).to_pydatetime()
            if convert_local_tz_to_utc:
                return local_tz.localize(timestamp).astimezone(UTC)
            elif convert_utc_to_local_tz:
                return timestamp.astimezone(local_tz)
            else:
                return timestamp

        elif isinstance(timestamp, dt.datetime):
            if convert_local_tz_to_utc:
                return local_tz.localize(timestamp).astimezone(UTC).strftime("%Y%m%d%H%M")
            elif convert_utc_to_local_tz:
                return timestamp.astimezone(local_tz).strftime("%Y%m%d%H%M")
            else:
                return timestamp.strftime("%Y%m%d%H%M")


class EntsoeClientError(Exception):

    def __init__(self, status_code: int, message: str, reason: str):
        self.status_code = status_code
        self.message = self._format_message(message)
        self.reason = reason

    def __str__(self) -> str:
        return f"\nStatus Code:\n\t{self.status_code} Client Error" \
               f"\nMessage:\n\t{self.message}" \
               f"\nReason:\n\t{self.reason}"

    @staticmethod
    def _format_message(message: str) -> str:
        start, stop = re.search(r"[0-9]+ Client Error: ", message).span()
        return message[0: start] + message[stop:]
