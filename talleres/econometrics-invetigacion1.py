import marimo

__generated_with = "0.2.8"
app = marimo.App(width="full")


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import re
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_spss('/tmp/Estructura CHC_2017.sav')
    relevantColumns = {
        'P23S1R': 'AHC',
        'P36R': 'Edad',
        'P28R': 'Edu',
        'P25': 'Redes',
        'P17': 'Salud',
        'P30S1': 'C_Cigarrillo',
        'P30S2': 'C_Alcohol',
        'P30S3': 'C_Marihuana',
        'P30S4': 'C_Inhalantes',
        'P30S5': 'C_Cocaina',
        'P30S6': 'C_Basuco',
        'P30S7': 'C_Heroina',
        'P30S8': 'C_Pepas',
        'P30S9': 'C_Otras',
        'COMPLETA': 'Completa'
    }


    #- set([ i for i in df.columns if re.search(r"(?i)P30S\d$",i) ])
    df = df[list(relevantColumns.keys())]
    return df, np, pd, plt, re, relevantColumns, sns


@app.cell
def __(df, relevantColumns):
    clean = df.rename(columns=relevantColumns)
    clean.info()
    return clean,


@app.cell
def __(clean, sns):
    sns.boxplot(data=clean,x='Edu',y='AHC',order=['Ninguno',
                                                  'Preescolar',
                                                  'Básica primaria',
                                                  'Básica secundaria', 
                                                  'Media académica, media técnica o normalista',
                                                  'Técnica profesional o tecnológica',
                                                  'Universitario o posgrado'])
    return


@app.cell
def __(clean, plt, sns):
    sns.boxplot(data=clean,x='Redes',y='AHC',order=['Ninguno', 'Hijo(a), hijastro(a)', 'Hermano(a), hermanastro(a)', 'Tío(a)', 'Otra', 'Pareja (esposo[a], compañero[a])', 'Mamá', 'Papá', 
                       'Abuelo(a)'])
    plt.gca().set_xlim(-1,9)
    plt.gca()
    return


@app.cell
def __(clean, sns):
    sns.boxplot(data=clean,x='C_Basuco', y='AHC')
    return


@app.cell
def __(clean, sns):
    sns.countplot(clean['Salud'])
    return


if __name__ == "__main__":
    app.run()
