from typing import Any, Dict, Union, Optional

import requests
from requests.exceptions import HTTPError
from bs4 import BeautifulSoup
import regex as re
import datetime as dt
import numpy as np
import pandas as pd

from gpg.data_retrieval.config import *


class EntsoeClient:

    def __init__(
            self,
            api_token: str,
            default_document_type: Optional[str] = "A65",
            default_process_type: Optional[str] = "A16",
            default_area: Optional[str] = "10YDE-VE-------2"
    ):
        self.api_token = api_token
        self.default_document_type = default_document_type
        self.default_process_type = default_process_type
        self.default_area = default_area
        self._base_url = BASE_URL

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

    def check_query_params(self, query_type) -> Dict[str, str]:
        """
        Returns required an optional query params for given query end point.
        """
        pass

    def default_query(self, start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
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

        return self.extract_data(
            response=self.query(
                document_type=self.default_document_type,
                process_type=self.default_process_type,
                area_code=self.default_area,
                start_date=start_date,
                end_date=end_date
            )
        )

    def query(
            self,
            document_type: str,
            process_type: str,
            area_code: str,  # bidding zone, area code or country code
            start_date: dt.datetime,
            end_date: dt.datetime,
            timeout: Optional[float] = 10.
    ) -> requests.Response:

        # TODO: self.check_query_params -> all the required params given?

        # Make query
        response = requests.get(
            url=self._base_url,
            params=dict(
                securityToken=self.api_token,
                documentType=document_type,
                processType=process_type,
                outBiddingZone_Domain=area_code,
                periodStart=self._format_timestamp(start_date),
                periodEnd=self._format_timestamp(end_date)
            ),
            timeout=timeout
        )

        # Verify query
        try:
            response.raise_for_status()
        except HTTPError as e:
            status_code = response.status_code
            message = e.args[0]
            reason = BeautifulSoup(response.text, "xml").find("text").get_text()
            raise EntsoeClientError(status_code=status_code, message=message, reason=reason) from None
        else:
            return response

    def extract_data(self, response: requests.Response) -> pd.DataFrame:
        soup = BeautifulSoup(response.text, "xml")
        dfs = []
        for ts in soup.find_all("TimeSeries"):
            start = self._format_timestamp(ts.start.get_text())
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

    def extract_query_details(self, response: requests.Response) -> Dict[str, Any]:
        soup = BeautifulSoup(response.text, "xml")
        return dict(
            document_type=soup.find("type").get_text(),
            # TODO: returns "A16" etc. -> convert this to corresponding doc type
            process_type=soup.find("process.processType").get_text(),  # TODO: same here.
            start_date=self._format_timestamp(soup.find("time_Period.timeInterval").start.get_text()),
            end_date=self._format_timestamp(soup.find("time_Period.timeInterval").end.get_text()),
            query_timestamp=self._format_timestamp(soup.find("createdDateTime").get_text())
        )

    @staticmethod
    def _format_timestamp(timestamp: Union[dt.datetime, str]) -> Union[str, dt.datetime]:
        if isinstance(timestamp, str):
            return pd.to_datetime(timestamp, yearfirst=True, utc=True).to_pydatetime()

        elif isinstance(timestamp, dt.datetime):
            return timestamp.strftime("%Y%m%d%H%M")

    def _get_area_code(self) -> str:
        pass


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
