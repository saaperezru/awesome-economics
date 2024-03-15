import marimo

__generated_with = "0.1.88"
app = marimo.App(width="full")


@app.cell
def __(exc):
    import altair as alt
    import marimo as mo

    import pandas as pd
    from pandas_datareader.base import _BaseReader


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

    df = dfOutput = OECDReader(symbols="OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE1_OUTPUT,1.0/A.EA19+NLD+DEU+GBR+BEL+NOR+CHE.......USD_EXC.V..",start='2018').read()
    dfExpenditure = OECDReader(symbols="OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_USD,1.0/A..EA20+NLD+NOR+COL.S13+S1M+S1..P7+P6+P51G+P3+B1GQ.....V..",start='2018').read()
    dfInflation =  OECDReader(symbols="OECD.SDD.TPS,DSD_PRICES@DF_PRICES_HICP,1.0/EA20+NLD.A.HICP..PA._T.N.GY",start='2018').read()
    return (
        OECDReader,
        OrderedDict,
        alt,
        df,
        dfExpenditure,
        dfInflation,
        dfOutput,
        itertools,
        mo,
        np,
        pd,
        re,
        read_jsdmx,
        sys,
    )


@app.cell
def __(mo):
    mo.md(r"""
    # Análisis de PIB e Inflación para Holanda
    [From 1st January 2015, the Euro area covers 19 countries: Austria, Belgium, Cyprus (*), Estonia, Finland, France, Germany, Greece, Ireland, Italy, Latvia, Lithuania, Luxembourg, Malta, the Netherlands, Portugal, Slovak Republic, Slovenia and Spain.](https://stats.oecd.org/OECDStat_Metadata/ShowMetadata.ashx?Dataset=NAAG&Coords=%5BLOCATION%5D.%5BEMU%5D&ShowOnWeb=true&Lang=en#:~:text=From%201st%20January%202015%2C%20the,Slovak%20Republic%2C%20Slovenia%20and%20Spain.)
    ## Distribución del PIB por el lado del gasto
    Tomando [datos](https://data-explorer.oecd.org/vis?lc=en&fs[0]=Topic%2C1%7CEconomy%23ECO%23%7CNational%20accounts%23ECO_NAD%23&fs[1]=Topic%2C2%7CEconomy%23ECO%23%7CNational%20accounts%23ECO_NAD%23%7CGDP%20and%20non-financial%20accounts%23ECO_NAD_GNF%23&pg=0&fc=Topic&snb=53&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_NAMAIN10%40DF_TABLE1_EXPENDITURE_GROWTH&df[ag]=OECD.SDD.NAD&df[vs]=1.0&pd=%2C&dq=A.NLD.S1..P41%2BP5%2BB11%2BP3T5%2BP3%2BB1GQ.......&ly[cl]=TIME_PERIOD&to[TIME_PERIOD]=false&lo=5&lom=LASTNPERIODS&vw=tb) de la OECD:
    """)
    return


@app.cell
def __(alt, dfExpenditure, mo, pd):
    def get_expenditure_data(df,country_name):
        dl = list(df.columns.names)
        dl.remove('Economic activity')
        dl.remove('Transaction')
        dl.remove('Reference area')
        
        dl.remove('Expenditure')
        dl.remove('Institutional sector')
        
        df.columns = df.columns.droplevel(dl)
        
        dfConsumption = df.xs(country_name,axis=1).xs('Final consumption expenditure',level='Transaction',axis=1).xs('Total',level='Expenditure',axis=1).xs('Not applicable', level='Economic activity',axis=1).drop(columns=['Total economy'])
        
        dfTotal = df.xs(country_name,axis=1).xs('Total economy',level='Institutional sector',axis=1).xs('Not applicable',level='Expenditure',axis=1)
        
        final = dfTotal.xs('Not applicable',level='Economic activity',axis=1).copy()
        final['Gross fixed capital formation'] = dfTotal.xs('Gross fixed capital formation', level='Transaction', axis=1)['Total - All activities']
        final = pd.concat([final, dfConsumption],axis=1)
        final = final.drop(columns=['Final consumption expenditure'])
        final['Net Export/Import'] = final['Exports of goods and services'] - final['Imports of goods and services']
        return final

    expData = []
    for i in dfExpenditure.columns.get_level_values('Reference area').drop_duplicates():
        countryData = get_expenditure_data(dfExpenditure,i)
        countryData = countryData.drop(columns=['Gross domestic product', 'Exports of goods and services', 'Imports of goods and services'])
        countryData = countryData.stack().reset_index()
        countryData.columns = ['Year','Economic activity','Dollars']
        countryData['Country']=i
        expData.append(countryData)
    expData=pd.concat(expData)
    expChart=alt.Chart(expData).mark_area().encode(
            x="Year:T",
            y = alt.Y("Dollars:Q")
            .title("Share of GDP")
            .stack("normalize")
            .axis(format=".0%"),
            color="Economic activity:N",
            facet="Country"
        ).properties(
        width=200,
        height=300,
    )
    mo.ui.altair_chart(expChart)
    return countryData, expChart, expData, get_expenditure_data, i


app._unparsable_cell(
    r"""
    netData = []
    #countryData.columns.drop(labels=['Gross domestic product', 'Exports of goods and services', 'Imports of goods and services'])
    for i2 in dfExpenditure.columns.get_level_values('Reference area').drop_duplicates():
        countryNetData = get_expenditure_data(dfExpenditure,i2)
        countryNetData = countryData.drop(columns=)
        countryNetData = countryNetData.stack().reset_index()
        countryNetData.columns = ['Year','Economic activity','Dollars']
        countryNetData['Country']=i2
        netData.append(countryData)
    netData=pd.concat(netData)
    expNetChart=alt.Chart(netData).mark_line().encode(
            x=\"Year:T\",
            y = alt.Y(\"Dollars:Q\")
            .title(\"Share of GDP\")
            .stack(\"normalize\")
            .axis(format=\".0%\"),
            color=\"Economic activity:N\"
        ).facet(
            facet=\"Country\", columns=2
        )
    mo.ui.altair_chart(expNetChart)
    """,
    name="__"
)


@app.cell
def __(mo):
    mo.md(r"""
    ##Distribución del PIB por el lado de la oferta
    Tomando [datos](https://data-explorer.oecd.org/vis?lc=en&fs[0]=Topic%2C1%7CEconomy%23ECO%23%7CNational%20accounts%23ECO_NAD%23&fs[1]=Topic%2C2%7CEconomy%23ECO%23%7CNational%20accounts%23ECO_NAD%23%7CGDP%20and%20non-financial%20accounts%23ECO_NAD_GNF%23&pg=0&fc=Topic&snb=53&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_NAMAIN10%40DF_TABLE1_OUTPUT&df[ag]=OECD.SDD.NAD&df[vs]=1.0&pd=%2C&dq=A.NLD.......USD_EXC.V..&ly[rw]=TRANSACTION%2CACTIVITY&ly[cl]=TIME_PERIOD&to[TIME_PERIOD]=false&lo=5&lom=LASTNPERIODS&vw=tb) de la OECD:
    """)
    return


@app.cell
def __(alt, dfOutput, mo, pd):
    def get_data(data,country_name):
        df = data.copy()
        drop_levels=list(df.columns.names)
        drop_levels.remove('Economic activity')
        drop_levels.remove('Transaction')
        drop_levels.remove('Reference area')
        df.columns = df.columns.droplevel(drop_levels)
        df = df.xs(country_name,level='Reference area',axis=1).xs("Value added, gross", axis=1)
        return df
    def get_country_data(data,country_name):
        df = get_data(data,country_name)
        df = df.drop(columns=['Not applicable','Total - All activities'])
        df = df.stack().reset_index()
        df.columns = ['Year','Economic activity','Dollars']
        return df
    def get_country_data_relative(data,country_name):
        df = get_data(data,country_name)
        df = df.apply(lambda x: x/df['Total - All activities'],axis=0)
        df = df.drop(columns=['Not applicable','Total - All activities'])
        df = df.stack().reset_index()
        df.columns = ['Year','Economic activity','Dollars']
        return df

    def get_mark_area_plot(data):
        import altair as alt
        chart=alt.Chart(data).mark_area().encode(
            x="Year:T",
            y = alt.Y("Dollars:Q")
            .title("Share of GDP")
            .stack("normalize")
            .axis(format=".0%"),
            color="Economic activity:N"
        )
        return mo.ui.altair_chart(chart)

    sortOrder = ['Agriculture, forestry and fishing',
                 'Construction',
                 'Public administration, defence, education, human health and social work activities',
                 'Professional, scientific and technical activities; administrative and support service activities',
                 'Wholesale and retail trade; repair of motor vehicles and motorcycles; transportation and storage; accommodation and food service activities',
                 'Manufacturing',     
                 'Industry (except construction)',
                 'Real estate activities',
                 'Financial and insurance activities',
                 'Arts, entertainment and recreation; other service activities; activities of household and extra-territorial organizations and bodies',
                'Information and communication']
    outputData = []
    for j in dfOutput.columns.get_level_values('Reference area').drop_duplicates():
        countryOData = get_country_data(dfOutput,j)
        countryOData['Country']=j
        outputData.append(countryOData)
    outputData=pd.concat(outputData)
    outChart=alt.Chart(outputData).mark_area().encode(
            x="Year:T",
            y = alt.Y("Dollars:Q")
            .title("Share of GDP")
            .stack("normalize")
            .axis(format=".0%"),
            color=alt.Color("Economic activity:N", sort=sortOrder),
            order=alt.Order('color_Economic activity_sort_index:Q'),
            facet="Country"
        ).properties(
        width=100,
        height=300,
    )
    mo.ui.altair_chart(outChart)
    return (
        countryOData,
        get_country_data,
        get_country_data_relative,
        get_data,
        get_mark_area_plot,
        j,
        outChart,
        outputData,
        sortOrder,
    )


@app.cell
def __(mo):
    mo.md(r"""
    ## Inflación
    Tomando [datos](https://data-explorer.oecd.org/vis?lc=en&tm=inflation&pg=0&hc[Measure]=Consumer%20price%20index&snb=32&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_PRICES%40DF_PRICES_HICP&df[ag]=OECD.SDD.TPS&df[vs]=1.0&pd=2018-01%2C&dq=NLD.A.HICP..PA._T.N.GY&ly[rw]=REF_AREA%2CMEASURE&ly[cl]=TIME_PERIOD&to[TIME_PERIOD]=false&vw=tb) de la OECD:""")
    return


@app.cell
def __(dfInflation):
    dfInflation.head()
    dfI = dfInflation.copy()
    dl2 = list(dfI.columns.names)
    dl2.remove('Measure')
    dl2.remove('Reference area')
    dfI.columns = dfI.columns.droplevel(dl2)
    dfI.xs('Consumer price index',level='Measure',axis=1).head()
    return dfI, dl2


if __name__ == "__main__":
    app.run()
