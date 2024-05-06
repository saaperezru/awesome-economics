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

    relevantColumns = {
        'P23S1R': 'AHC',
        'P1': 'Departamento',
        'P1S1': 'Municipio',
        'P28R': 'Edu',
        'P35': 'Sexo',
        'P22': 'MotivoInicio',
        'P24': 'MotivoPermanencia',
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
        'P36R': 'EdadEstimada',
        'COMPLETA': 'Completa'
    }

    chc2017 = pd.read_spss('/tmp/chc/2017.sav')
    relevantColumns2017 = relevantColumns.copy()
    relevantColumns2017['P8'] = 'Edad'
    relevantColumns2017['P9R'] = 'Genero'
    chc2017 = chc2017[list(relevantColumns2017.keys())]
    chc2017 = chc2017.rename(columns=relevantColumns2017)

    chc2019 = pd.read_csv('/tmp/chc/2019.csv', sep=';')
    relevantColumns2019 = relevantColumns.copy()
    relevantColumns2019['P28R'] = 'Edad'
    relevantColumns2019['P9'] = 'Genero'
    chc2019 = chc2019[list(relevantColumns2019.keys())]
    chc2019 = chc2019.rename(columns=relevantColumns2019)

    chc2020 = pd.read_spss('/tmp/chc/2020.sav')
    chc2020 = chc2020[list(relevantColumns2019.keys())]
    chc2020 = chc2020.rename(columns=relevantColumns2019)

    chc2021 = pd.read_spss('/tmp/chc/2021.sav')
    chc2021 = chc2021[list(relevantColumns2019.keys())]
    chc2021 = chc2021.rename(columns=relevantColumns2019)
    return (
        chc2017,
        chc2019,
        chc2020,
        chc2021,
        np,
        pd,
        plt,
        re,
        relevantColumns,
        relevantColumns2017,
        relevantColumns2019,
        sns,
    )


@app.cell
def __(chc2017, chc2019, chc2020, chc2021, pd):
    clean = pd.concat([chc2017,chc2019,chc2020,chc2021])
    #clean = clean[clean["Edad"].between(14,28)]
    #clean = clean[clean["MotivoPermanencia"]!='Siempre ha vivido en la calle']
    #clean = clean[clean["MotivoInicio"]!='Siempre ha vivido en la calle']
    clean['SPA'] = (clean['C_Cigarrillo']=='Sí') | (clean['C_Alcohol']=='Sí' ) | (clean['C_Marihuana']=='Sí' ) | (clean['C_Inhalantes']=='Sí' ) | (clean['C_Cocaina']=='Sí' ) | (clean['C_Basuco']=='Sí' ) | (clean['C_Heroina']=='Sí' ) | (clean['C_Pepas']=='Sí' ) | (clean['C_Otras']=='Sí' )
    clean['PVHC']=clean['AHC']/clean['Edad']
    clean.info()
    #clean.to_stata('/tmp/chc/all.dta')
    return clean,


@app.cell
def __(clean):
    clean.describe()
    return


@app.cell
def __(clean, plt):
    clean.hist()
    plt.gca()
    return


@app.cell
def __(clean):
    clean['SPA'].value_counts()
    return


@app.cell
def __(clean, sns):
    sns.lmplot(data=clean, x="AHC",y="Edad")
    return


@app.cell
def __(clean, plt, sns):
    sns.boxplot(data=clean,x='Edu',y='AHC',order=['Ninguno',
                                                  'Preescolar',
                                                  'Básica primaria',
                                                  'Básica secundaria', 
                                                  'Media académica, media técnica o normalista',
                                                  'Técnica profesional o tecnológica',
                                                  'Universitario o posgrado'])
    plt.xticks(rotation=90)
    plt.gca()
    return


@app.cell
def __(clean, plt, sns):
    sns.boxplot(data=clean,x='Salud',y='AHC')
    plt.xticks(rotation=90)
    plt.gca()
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
    sns.boxplot(data=clean,x='SPA', y='AHC')
    return


@app.cell
def __(clean, plt, sns):
    sns.boxplot(data=clean,x='MotivoInicio', y='PVHC')
    #plt.figure(figsize=(3,4))
    plt.xticks(rotation=90)
    plt.gca()
    return


@app.cell
def __(clean, sns):
    sns.countplot(clean['Salud'])
    return


@app.cell
def __(clean, plt, sns):
    sns.stripplot(data=clean,x='AHC',y='MotivoInicio')
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
