import marimo

__generated_with = "0.1.88"
app = marimo.App()


@app.cell
def __(exc):
    import pandas as pd

    from collections import OrderedDict
    import itertools
    import re
    import sys

    import numpy as np
    import pandas as pd

    from pandas_datareader.io.util import _read_content

    def _get_indexer(index):
        if index.nlevels == 1:
            return [str(i) for i in range(len(index))]
        else:
            it = itertools.product(*[range(len(level)) for level in index.levels])
            return [":".join(map(str, i)) for i in it]


    def _fix_quarter_values(value):
        """Make raw quarter values Pandas-friendly (e.g. 'Q4-2018' -> '2018Q4')."""
        m = re.match(r"Q([1-4])-(\d\d\d\d)", value)
        if not m:
            return value
        quarter, year = m.groups()
        value = f"{quarter}Q{year}"
        return value


    def _parse_values(dataset, index, columns):
        size = len(index)
        series = dataset["series"]

        values = []
        # for s_key, s_value in iteritems(series):
        for s_key in _get_indexer(columns):
            try:
                observations = series[s_key]["observations"]
                observed = []
                for o_key in _get_indexer(index):
                    try:
                        observed.append(observations[o_key][0])
                    except KeyError:
                        observed.append(np.nan)
            except KeyError:
                observed = [np.nan] * size

            values.append(observed)

        return np.transpose(np.array(values))


    def _parse_dimensions(dimensions):
        arrays = []
        names = []
        for key in dimensions:
            values = list(filter(lambda x: "name" in x.keys(), key["values"]))
            values = [v["name"] for v in values]

            role = key.get("role", None)
            if role in ("time", "TIME_PERIOD"):
                values = [_fix_quarter_values(v) for v in values]
                values = pd.DatetimeIndex(values)

            arrays.append(values)
            names.append(key["name"])
        midx = pd.MultiIndex.from_product(arrays, names=names)
        if len(arrays) == 1 and isinstance(midx, pd.MultiIndex):
            # Fix for pandas >= 0.21
            midx = midx.levels[0]

        return midx


    def read_jsdmx(path_or_buf):
        """
        Convert a SDMX-JSON string to pandas object

        Parameters
        ----------
        path_or_buf : a valid SDMX-JSON string or file-like
            https://github.com/sdmx-twg/sdmx-json

        Returns
        -------
        results : Series, DataFrame, or dictionary of Series or DataFrame.
        """

        jdata = _read_content(path_or_buf)

        try:
            import simplejson as json
        except ImportError as exc:
            if sys.version_info[:2] < (2, 7):
                raise ImportError("simplejson is required in python 2.6") from exc
            import json

        if isinstance(jdata, dict):
            data = jdata
        else:
            data = json.loads(jdata, object_pairs_hook=OrderedDict)

        structure = data["data"]["structures"][0]
        index = _parse_dimensions(structure["dimensions"]["observation"])
        columns = _parse_dimensions(structure["dimensions"]["series"])

        dataset = data["data"]["dataSets"]
        if len(dataset) != 1:
            raise ValueError("length of 'dataSets' must be 1")
        dataset = dataset[0]
        values = _parse_values(dataset, index=index, columns=columns)

        df = pd.DataFrame(values, columns=columns, index=index)
        return df


    class OECDReader(_BaseReader):
        """Get data for the given name from OECD."""

        _format = "json"

        @property
        def url(self):
            """API URL"""
            url = "https://sdmx.oecd.org/public/rest/data"

            if not isinstance(self.symbols, str):
                raise ValueError("data name must be string")

            print(f"{url}/{self.symbols}/?format=jsondata{'&startPeriod='+self.start.isoformat() if self.start else ''}")
            return f"{url}/{self.symbols}/?format=jsondata{'&startPeriod='+self.start.isoformat() if self.start else ''}"

        def _read_lines(self, out):
            """read one data from specified URL"""
            df = read_jsdmx(out)
            try:
                idx_name = df.index.name  # hack for pandas 0.16.2
                df.index = pd.to_datetime(df.index, errors="ignore")
                for col in df:
                    df[col] = pd.to_numeric(df[col], errors="ignore")
                df = df.sort_index()
                df = df.truncate(self.start, self.end)
                df.index.name = idx_name
            except ValueError:
                pass
            return df
            
    df = OECDReader(symbols="OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE1_OUTPUT,1.0/A.NLD+COL+DEU+USA.......USD_EXC.V..",start='2014').read()
    return (
        OECDReader,
        OrderedDict,
        df,
        itertools,
        np,
        pd,
        re,
        read_jsdmx,
        sys,
    )


@app.cell
def __(df):
    import marimo as mo
    def get_data(data,country_name):
        df = data.copy()
        drop_levels=list(df.columns.names)
        drop_levels.remove('Economic activity')
        drop_levels.remove('Transaction')
        drop_levels.remove('Reference area')
        df.columns = df.columns.droplevel(drop_levels)
        df = df.xs(country_name,axis=1).xs("Value added, gross", axis=1)
        return df
    def get_country_data(data,country_name):
        df = get_data(data,country_name)
        df = df.drop(columns=['Not applicable','Total - All activities'])
        df = df.stack().reset_index()
        df.columns = ['Year','Economic activity','Dollars']
        return df
    def get_country_data_relative(data,country_name):
        df = get_data(data,country_name)
        df = df.drop(columns=['Not applicable','Total - All activities'])
        nldp=nld.apply(lambda x: x/nld['Total - All activities'])
        nldp = nldp.drop(columns=['Total - All activities'])
        nldp.head()
        df = df.stack().reset_index()
        df.columns = ['Year','Economic activity','Dollars']
        return df

    def get_mark_area_plot(data):
        import altair as alt
        chart=alt.Chart(data).mark_area().encode(
            x="Year:T",
            y="Dollars:Q",
            color="Economic activity:N"
        )
        return mo.ui.altair_chart(chart)

    colo = get_mark_area_plot(get_country_data(df,"Colombia"))
    nld = get_mark_area_plot(get_country_data(df,"Netherlands"))
    return (
        colo,
        get_country_data,
        get_country_data_relative,
        get_data,
        get_mark_area_plot,
        mo,
        nld,
    )


@app.cell
def __(colo, mo):
    mo.vstack([colo, colo.value.head()])
    return


@app.cell
def __(mo, nld):
    mo.vstack([nld, nld.value.head()])
    return


@app.cell
def __():
    return


@app.cell
def __(df3):
    col=df3.xs("Colombia",axis=1).xs("Value added, gross", axis=1)
    #col["Country"]="COL"
    col.head()
    return col,


if __name__ == "__main__":
    app.run()
