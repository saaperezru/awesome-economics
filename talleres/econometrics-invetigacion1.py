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

    import marimo as mo
    from biokit.viz import corrplot
    import seaborn as sns
    from scipy import stats

    relevantColumns = {
        'P23S1R': 'AHC',
        'P1': 'Departamento',
        'P1S1': 'Municipio',
        'P28R': 'Edu',
        'P22': 'MotivoInicio',
        'P24': 'MotivoPermanencia',
        'P25': 'Redes',
        'P17': 'Salud',
        'P26S3': 'Ayuda_Oficial',
        'P16S1': 'A_Oir',
        'P16S2': 'A_Hablar',
        'P16S3': 'A_Ver',
        'P16S4': 'A_Mover',
        'P16S5': 'A_Agarrar',
        'P16S6': 'A_Aprender',
        'P16S7': 'A_ComerVestir',
        'P16S8': 'A_Relacionarse',
        'P16S9': 'A_CardiacoRespiratorio',
        'P30S1': 'C_Cigarrillo',
        'P30S2': 'C_Alcohol',
        'P30S3': 'C_Marihuana',
        'P30S4': 'C_Inhalantes',
        'P30S5': 'C_Cocaina',
        'P30S6': 'C_Basuco',
        'P30S7': 'C_Heroina',
        'P30S8': 'C_Pepas',
        'P30S9': 'C_Otras',
        'P30_1': 'C_Principal',
        'P30S1A1': 'EI_Cigarrillo',
        'P30S2A1': 'EI_Alcohol',
        'P30S3A1': 'EI_Marihuana',
        'P30S4A1': 'EI_Inhalantes',
        'P30S5A1': 'EI_Cocaina',
        'P30S6A1': 'EI_Basuco',
        'P30S7A1': 'EI_Heroina',
        'P30S8A1': 'EI_Pepas',
        'P30S9A1': 'EI_Otras',
        'P36R': 'EdadEstimada',
        'COMPLETA': 'Completa'
    }

    chc2017 = pd.read_spss('/tmp/chc/2017.sav')
    relevantColumns2017 = relevantColumns.copy()
    relevantColumns2017['P8'] = 'Edad'
    relevantColumns2017['P9R'] = 'Genero'
    chc2017 = chc2017[list(relevantColumns2017.keys())]
    chc2017 = chc2017.rename(columns=relevantColumns2017)
    chc2017['Año']='2017'

    chc2019 = pd.read_csv('/tmp/chc/2019.csv', sep=';')
    relevantColumns2019 = relevantColumns.copy()
    relevantColumns2019['P8R'] = 'Edad'
    relevantColumns2019['P9'] = 'Genero'
    chc2019 = chc2019[list(relevantColumns2019.keys())]
    chc2019 = chc2019.rename(columns=relevantColumns2019)
    chc2019['Año']='2019'
    chc2019['Departamento'] = chc2019['Departamento'].astype('float')

    chc2020 = pd.read_spss('/tmp/chc/2020.sav')
    chc2020 = chc2020[list(relevantColumns2019.keys())]
    chc2020 = chc2020.rename(columns=relevantColumns2019)
    chc2020['Año']='2020'
    Edu2020Map = {
        8:13,
        7:9,
        6:7
    }
    chc2020['Edu']=chc2020['Edu'].replace(Edu2020Map)
    chc2020['Departamento'] = chc2020['Departamento'].astype('float')


    for k in list(relevantColumns2019.keys()):
        if(re.match('P30S.A1',k)):
            relevantColumns2019[k+'R'] = relevantColumns[k]
            del relevantColumns2019[k]
    chc2021 = pd.read_csv('/tmp/chc/2021.csv')
    chc2021 = chc2021[list(relevantColumns2019.keys())]
    chc2021 = chc2021.rename(columns=relevantColumns2019)
    chc2021['Edu']=chc2021['Edu'].replace(Edu2020Map)
    chc2021['Año']='2021'
    chc2021['Departamento'] = chc2021['Departamento'].astype('float')
    return (
        Edu2020Map,
        chc2017,
        chc2019,
        chc2020,
        chc2021,
        corrplot,
        k,
        mo,
        np,
        pd,
        plt,
        re,
        relevantColumns,
        relevantColumns2017,
        relevantColumns2019,
        sns,
        stats,
    )


@app.cell
def __(chc2019, chc2020, chc2021, pd):
    clean = pd.concat([chc2019,chc2020,chc2021],ignore_index=True)
    clean['Departamento'] = clean['Departamento'].astype('category')
    #clean['Departamento'].replace(to_replace=[None], value="", inplace=True)
    clean = clean[clean['EdadEstimada'].isna()]
    clean.info()
    #clean.to_stata('/tmp/chc/all.dta')
    return clean,


@app.cell
def __(clean):
    clean.describe()
    return


@app.cell
def __(clean):
    clean.info()
    return


@app.cell
def __(np, pd, plt, sns, stats):
    def plot2(x, y, **kws):
        CONTINUOUS_TYPES =  ['int16','int8','float64']
        if x.dtype=='category' and y.dtype == 'category':
            sns.heatmap(pd.crosstab(x,y,normalize=True).transpose(),annot=True, fmt=".1f",cmap="crest",cbar=False,**kws)
            #g.set_yticklabels(labels=g.get_yticklabels(), va='center')

    def plot(x, y, **kws):
        #print(x.dtype,y.dtype)
        CONTINUOUS_TYPES =  ['int16','int8','float']
        if y.dtype=='category' and x.dtype in CONTINUOUS_TYPES:
            sns.boxplot(x=x,y=y,**kws)
        elif x.dtype=='category' and y.dtype in CONTINUOUS_TYPES:
            sns.boxplot(x=x,y=y,**kws)
            #stats.ttest_ind(y, x)
        elif y.dtype in CONTINUOUS_TYPES and x.dtype in CONTINUOUS_TYPES:
            #corrfunc(x,y)
            pearsonsig(x,y)
            sns.regplot(x=x,y=y,line_kws={"color": "red"},**kws)
            confidence_ellipse(x,y)

    def test(x, **kws):
        if x.dtype=='category':
            sns.barplot(x=x.value_counts().index
                ,y=x.value_counts())
        else:
            sns.histplot(x,kde=True,**kws)


    def dist_corr(X, Y, pval=False, nruns=2000):
        """ Distance correlation with p-value from bootstrapping
        """
        import dcor
        dc = dcor.distance_correlation(X, Y)
        if pval:
            pv = dcor.independence.distance_covariance_test(X, Y, exponent=1.0, num_resamples=nruns).pvalue
            return (dc, pv)
        else:
            return (dc,1)

    def corrfunc(x, y, **kws):
        d = pd.DataFrame([x.reset_index(drop=True),y.reset_index(drop=True)]).T.dropna()
        d, p = dist_corr(d.iloc[:,0],d.iloc[:,1])
        #print("{:.4f}".format(d), "{:.4f}".format(p))
        if p > 0.01:
            star=False
            pclr = 'pink'
        else:
            star=True
            pclr= 'maroon'
        ax = plt.gca()
        ax.annotate("DC = {:.2f}{}".format(d,'*' if star else ''), xy=(.7, 0.8), xycoords=ax.transAxes, color = pclr, fontsize = 10)

    def pearsonsig(x, y, **kws):
        d = pd.DataFrame([x.reset_index(drop=True),y.reset_index(drop=True)]).T.dropna()
        c, p = stats.pearsonr(d.iloc[:,0],d.iloc[:,1])
        if p > 0.01:
            pclr = 'pink'
            star=False
        else:
            pclr= 'maroon'
            star=True
        ax = plt.gca()
        ax.annotate("PC = {:.2f}{}".format(c,'*' if star else ''), xy=(.7, 0.9), xycoords=ax.transAxes, color = pclr, fontsize = 10)

    #Taken from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    def confidence_ellipse(x, y, n_std=2.0, **kwargs):
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.

        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input data.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`

        Returns
        -------
        matplotlib.patches.Ellipse
        """
        ax = plt.gca()
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, fill=False, **kwargs)

        # Calculating the standard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the standard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def hide_current_axis(*args, **kwds):
        plt.gca().set_visible(False)

    def plot_variables(data,columns):
        d = data[columns]
        d.replace([np.inf, -np.inf, np.nan], None)
        g = sns.PairGrid(d, vars=columns)
        g.map_diag(test)
        g.map_lower(plot2)
        g.map_lower(plot)
        plt.subplots_adjust(left=0.35,bottom=0.35)
        return g

    def autonomy_boolean(column):
        r = column=="Sin dificultad"
        r[column.isna()] = np.nan
        return r.astype('category')
    return (
        autonomy_boolean,
        confidence_ellipse,
        corrfunc,
        dist_corr,
        hide_current_axis,
        pearsonsig,
        plot,
        plot2,
        plot_variables,
        test,
    )


@app.cell
def __(autonomy_boolean, clean, np, pd):
    analysisData = clean[clean['MotivoInicio']!=9]
    analysisData['PVHC']=analysisData['AHC']/analysisData['Edad']
    analysisData['Año']=analysisData['Año'].astype('category')
    analysisData['Edad2'] = analysisData['Edad']*analysisData['Edad']
    analysisData['LEdad'] = np.log(analysisData['Edad'])
    analysisData['Departamento'] = analysisData['Departamento'].astype('category')
    DepToRegionsMap = {
    5:      "Andina",
    8:      "Caribe",
    13:     "Caribe",
    15:     "Andina",
    17:     "Andina",
    18:     "Amazonia",
    19:     "Pacifico",
    20:     "Caribe",
    23:     "Caribe",
    25:     "Andina",
    27:     "Pacifico",
    41:     "Andina",
    44:     "Caribe",
    47:     "Caribe",
    50:     "Orinoquia",
    52:     "Pacifico",
    54:     "Andina",
    63:     "Andina",
    66:     "Andina",
    68:     "Andina",
    70:     "Caribe",
    73:     "Andina",
    76:     "Pacifico",
    81:     "Orinoquia",
    85:     "Orinoquia",
    86:     "Amazonia",
    91:     "Amazonia",
    99:     "Orinoquia"
    }
    analysisData['Region'] = analysisData['Departamento'].replace(DepToRegionsMap).astype('category')
    analysisData['Municipio'] = analysisData['Municipio'].astype('category')
    GeneroCategories = {
        1: 'Hombre',
        2: 'Mujer'
    }
    analysisData['Genero'] = analysisData['Genero'].astype('category').cat.rename_categories(GeneroCategories)
    MotivoInicioCategories = {
        1:"Consumo de sustancias psicoactivas",
        2: "Por gusto personal",
        3:	"Amenaza o riesgo para su vida o integridad física",
        4:	"Influencia de otras personas",
        5:	"Dificultades económicas",
        6:	"Falta de trabajo",
        7:	"Conflictos o dificultades familiares",
        8:	"Abuso sexual",
        9:	"Siempre ha vivido en la calle",
        10:	"Victima del conflicto armado o desplazado",
        11:	"Otra"
    }
    MotivoInicioGroups = {
        1:  0,
        2:  2,
        3:	1,
        4:	2,
        5:	3,
        6:	3,
        7:	2,
        8:	2,
        9:	5,
        10:	1,
        11:	1
    }
    MotivoInicioGroupsNames = {
        0: "Consumo de sustancias psicoactivas",
        1: "Otras",
        2: "Conflictos/Abusos intererpersonales",
        3: "Economicas"
    }
    analysisData['MotivoInicioGroups'] = analysisData['MotivoInicio'].replace(MotivoInicioGroups).astype('category').cat.rename_categories(MotivoInicioGroupsNames)
    analysisData['MotivoInicio'] = analysisData['MotivoInicio'].astype('category').cat.rename_categories(MotivoInicioCategories)
    MotivoPermanenciaCategories = {
        1: "Consumo de sustancias psicoactivas",
        2: "Por gusto personal",
        3: "Las amistades",
        4: "Dificultades económicas",
        5: "Falta de trabajo",
        6: "Enfermedad",
        7: "Conflictos o dificultades familiares",
        8: "Siempre ha vivido en la calle",
        9: "Soledad",
        10: "Está haciendo proceso en un centro de atención",
        11: "Otra"
    }
    analysisData['MotivoPermanencia'] = analysisData['MotivoPermanencia'].astype('category').cat.rename_categories(MotivoPermanenciaCategories)
    RedesCategories = {
        1: "Mamá",
        2: "Papá",
        3: "Hermano(a), hermanastro(a)",
        4: "Abuelo(a)",
        5: "Tío(a)",
        6: "Hijo(a), hijastro(a)",
        7: "Pareja (esposo[a], compañero[a])",
        8: "Otra",
        9: "Ninguno"
    }
    RedesGroupsMap = {
        1: 1,
        2: 1,
        3: 1,
        4:  0,
        5:  0,
        6: 2,
        7: 2,
        8:  0,
        9:  3
    }
    RedesGroupsMapNames = {
        0: "Familia extendida",
        1: "Primera infancia/Padres",
        2: "Familia nuclear",
        3: "Ninguno"
    }
    analysisData['RedesGroups'] = analysisData['Redes'].replace(RedesGroupsMap).astype('category').cat.rename_categories(RedesGroupsMapNames)
    analysisData['Redes'] = analysisData['Redes'].astype('category').cat.rename_categories(RedesCategories)


    CPrincipalMap={
        1:"Cigarrillo",
        2:"Alcohol",
        3:"Marihuana",
        4:"Inhalantes",
        5:"Cocaina",
        6:"Basuco",
        7:"Heroina",
        8:"Pepas",
        9:"Otras"
    }
    CPrincipalGroupsMap={
        1: 0,
        2: 1,
        3: 2,
        4:  0,
        5:  0,
        6:  0,
        7:  1,
        8:  2,
        9:  0
    }
    CPrincipalGroupsName={
        0: "Estimulantes",
        1: "Depresor",
        2: "Alucinogenas"
    }
    analysisData['C_PrincipalGroups'] = analysisData['C_Principal'].replace(CPrincipalGroupsMap).astype('category').cat.rename_categories(CPrincipalGroupsName)
    analysisData['C_Principal'] = analysisData['C_Principal'].astype('category').cat.rename_categories(CPrincipalMap)
    analysisData['C_Cigarrillo'] = analysisData['C_Cigarrillo'].astype('category')
    analysisData['C_Alcohol'] = analysisData['C_Alcohol'].astype('category')
    analysisData['C_Marihuana'] = analysisData['C_Marihuana'].astype('category')
    analysisData['C_Inhalantes'] = analysisData['C_Inhalantes'].astype('category')
    analysisData['C_Cocaina'] = analysisData['C_Cocaina'].astype('category')
    analysisData['C_Basuco'] = analysisData['C_Basuco'].astype('category')
    analysisData['C_Pepas'] = analysisData['C_Pepas'].astype('category')
    analysisData['C_Heroina'] = analysisData['C_Heroina'].astype('category')
    analysisData['C_Otras'] = analysisData['C_Otras'].astype('category')

    analysisData['PVC_Cigarrillo'] = (analysisData['Edad'] - analysisData['EI_Cigarrillo'])/analysisData['Edad']
    analysisData['PVC_Alcohol'] = (analysisData['Edad'] - analysisData['EI_Alcohol'])/analysisData['Edad']
    analysisData['PVC_Marihuana'] = (analysisData['Edad'] - analysisData['EI_Marihuana'])/analysisData['Edad']
    analysisData['PVC_Inhalantes'] = (analysisData['Edad'] - analysisData['EI_Inhalantes'])/analysisData['Edad']
    analysisData['PVC_Cocaina'] = (analysisData['Edad'] - analysisData['EI_Cocaina'])/analysisData['Edad']
    analysisData['PVC_Basuco'] = (analysisData['Edad'] - analysisData['EI_Basuco'])/analysisData['Edad']
    analysisData['PVC_Pepas'] = (analysisData['Edad'] - analysisData['EI_Pepas'])/analysisData['Edad']
    analysisData['PVC_Heroina'] = (analysisData['Edad'] - analysisData['EI_Heroina'])/analysisData['Edad']
    analysisData['PVC_Otras'] = (analysisData['Edad'] - analysisData['EI_Otras'])/analysisData['Edad']

    analysisData['PVC_Baja'] = pd.DataFrame([analysisData['PVC_Cigarrillo'],analysisData['PVC_Alcohol'],analysisData['PVC_Marihuana']]).max()
    analysisData['PVC_Media'] = pd.DataFrame([analysisData['PVC_Inhalantes'],analysisData['PVC_Basuco'],analysisData['PVC_Otras']]).max()
    analysisData['PVC_Alta'] = pd.DataFrame([analysisData['PVC_Pepas'],analysisData['PVC_Heroina'],analysisData['PVC_Cocaina']]).max()
    analysisData['PVC_Alguna'] = pd.DataFrame([analysisData['PVC_Cigarrillo'],analysisData['PVC_Alcohol'],analysisData['PVC_Marihuana'],analysisData['PVC_Inhalantes'],analysisData['PVC_Basuco'],analysisData['PVC_Otras'],analysisData['PVC_Pepas'],analysisData['PVC_Heroina'],analysisData['PVC_Cocaina']]).max()

    ACategories = {
        1: 'No puede hacerlo',
        2: 'Sí, con mucha dificultad',
        3: 'Sí, con alguna dificultad',
        4: 'Sin dificultad'
    }
    analysisData['A_Oir'] = analysisData['A_Oir'].astype('category').cat.rename_categories(ACategories)
    analysisData['AB_Oir'] = autonomy_boolean(analysisData['A_Oir'])
    analysisData['A_Hablar'] = analysisData['A_Hablar'].astype('category').cat.rename_categories(ACategories)
    analysisData['AB_Hablar'] = autonomy_boolean(analysisData['A_Hablar'])
    analysisData['A_Ver'] = analysisData['A_Ver'].astype('category').cat.rename_categories(ACategories)
    analysisData['AB_Ver'] = autonomy_boolean(analysisData['A_Ver'])
    analysisData['A_Mover'] = analysisData['A_Mover'].astype('category').cat.rename_categories(ACategories)
    analysisData['AB_Mover'] = autonomy_boolean(analysisData['A_Mover'])
    analysisData['A_Agarrar'] = analysisData['A_Agarrar'].astype('category').cat.rename_categories(ACategories)
    analysisData['AB_Agarrar'] = autonomy_boolean(analysisData['A_Agarrar'])
    analysisData['A_Aprender'] = analysisData['A_Aprender'].astype('category').cat.rename_categories(ACategories)
    analysisData['AB_Aprender'] = autonomy_boolean(analysisData['A_Aprender'])
    analysisData['A_ComerVestir'] = analysisData['A_ComerVestir'].astype('category').cat.rename_categories(ACategories)
    analysisData['AB_ComerVestir'] = autonomy_boolean(analysisData['A_ComerVestir'])
    analysisData['A_Relacionarse'] = analysisData['A_Relacionarse'].astype('category').cat.rename_categories(ACategories)
    analysisData['AB_Relacionarse'] = autonomy_boolean(analysisData['A_Relacionarse'])
    analysisData['A_CardiacoRespiratorio'] = analysisData['A_CardiacoRespiratorio'].astype('category').cat.rename_categories(ACategories)
    analysisData['AB_CardiacoRespiratorio'] = autonomy_boolean(analysisData['A_CardiacoRespiratorio'])

    EduCategories = {
        1: "Preescolar",
        2: "Básica primaria",
        3: "Básica secundaria",
        4: "Media académica clásica",
        5: "Media técnica",
        6: "Normalista",
        7: "Técnica profesional",
        8: "Tecnológica",
        9: "Universitario",
        10: "Especialización",
        11: "Maestria",
        12: "Doctorado",
        13: "Ninguno"
    }
    EduGroupsMapping = {
        1: 0,
        2: 1,
        3: 1,
        4: 1,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 2,
        11: 2,
        12: 2,
        13: 0
    }
    EduGroupsCategories = {
        0: "Ninguno",
        1: "Colegio",
        2: "Profesional"
    }
    analysisData['EduGroups'] = analysisData['Edu'].replace(EduGroupsMapping).astype('category').cat.rename_categories(EduGroupsCategories)
    analysisData['Edu'] = analysisData['Edu'].astype('category').cat.rename_categories(EduCategories)
    analysisData['Completa'] = analysisData['Completa'].astype('category')

    analysisData.info()
    return (
        ACategories,
        CPrincipalGroupsMap,
        CPrincipalGroupsName,
        CPrincipalMap,
        DepToRegionsMap,
        EduCategories,
        EduGroupsCategories,
        EduGroupsMapping,
        GeneroCategories,
        MotivoInicioCategories,
        MotivoInicioGroups,
        MotivoInicioGroupsNames,
        MotivoPermanenciaCategories,
        RedesCategories,
        RedesGroupsMap,
        RedesGroupsMapNames,
        analysisData,
    )


@app.cell
def __(analysisData):
    analysisData['RedesGroups'].value_counts()
    return


@app.cell
def __(analysisData, plt, sns):
    sns.boxplot(analysisData, x='Edu', y='PVHC', hue='EduGroups')
    plt.xticks(rotation=90)
    plt.gca()
    return


@app.cell
def __(analysisData, clean, sns):
    analysisData.to_stata('/tmp/chc/analysis.dta')

    -    sns.boxplot(data=clean,x='Edu',y='AHC')


    return


@app.cell
def __(analysisData, mo):
    import statsmodels.formula.api as smf

    formula = "PVHC ~ C(Region,Treatment('Orinoquia')) + C(Año,Treatment('2019')) + C(RedesGroups, Treatment('Ninguno')) + C(EduGroups, Treatment('Ninguno')) + C(Genero, Treatment('Hombre')) + PVC_Alguna + C(MotivoInicioGroups,Treatment('Consumo de sustancias psicoactivas'))  + Edad + Edad2 + C(C_PrincipalGroups,Treatment('Depresor')) + 1"
    mod = smf.ols(formula, data=analysisData)
    res = mod.fit()
    mo.md(f"""
    {mo.as_html(res.summary())}
    """)
    return formula, mod, res, smf


@app.cell
def __(res):
    from statsmodels.stats.diagnostic import het_white
    #perform White's test
    white_test = het_white(res.resid,  res.model.exog)

    #define labels to use for output of White's test
    labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']

    #print results of White's test
    dict(zip(labels, white_test))
    return het_white, labels, white_test


@app.cell
def __(analysisData, pd):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # VIF dataframe 
    vif_data = pd.DataFrame() 
    vif_data["feature"] = analysisData.columns 

    # calculating VIF for each feature 
    vif_data["VIF"] = [variance_inflation_factor(analysisData.values, i) 
                              for i in range(len(analysisData.columns))] 

    print(vif_data)
    return variance_inflation_factor, vif_data


@app.cell
def __(analysisData, mo, smf):
    formula3 = "PVHC ~ C(Redes, Treatment('Ninguno')) + C(Edu, Treatment('Ninguno')) + PVC_Alguna + C(MotivoInicio,Treatment('Siempre ha vivido en la calle')) + C(Genero, Treatment('Mujer')) + 1"
    mod3 = smf.ols(formula3, data=analysisData[analysisData['MotivoPermanencia']=='Por gusto personal'])
    res3 = mod3.fit()
    mo.md(f"""
    {mo.as_html(res3.summary())}
    """)
    return formula3, mod3, res3


@app.cell
def __():
    #plot_variables(analysisData, ['PVHC','Edad'])
    return


@app.cell
def __():
    #plot_variables(analysisData, ['PVHC','MotivoInicio','MotivoPermanencia'])
    return


@app.cell
def __(analysisData, mo, smf):
    formula2 = "PVHC ~ PVC_Cigarrillo + PVC_Alcohol+ PVC_Marihuana + PVC_Inhalantes + PVC_Cocaina + PVC_Basuco + PVC_Heroina + PVC_Pepas +PVC_Otras"
    mod2 = smf.ols(formula2, data=analysisData[analysisData['MotivoPermanencia']=='Consumo de sustancias psicoactivas'])
    res2 = mod2.fit()
    mo.md(f"""
    {mo.as_html(res2.summary())}
    """)
    return formula2, mod2, res2


@app.cell
def __(plt, res):
    from statsmodels.graphics.regressionplots import plot_regress_exog
    fig=plot_regress_exog(res, 'PVC_Baja')
    fig.tight_layout(pad=1.0)
    plt.show()
    return fig, plot_regress_exog


@app.cell
def __(analysisData, plt, res):
    #Get the predicted values of y from the fitted model
    y_cap = res.predict(analysisData)

    #Plot the model's residuals against the predicted values of y
    plt.xlabel('Predicted value of Percent_Households_Below_Poverty_Level')

    plt.ylabel('Residual error')

    plt.scatter(y_cap, res.resid)
    return y_cap,


@app.cell
def __():
    #plot_variables(analysisData, ['PVHC', 'PVC_Baja', 'PVC_Media', 'PVC_Alta'])
    return


@app.cell
def __(analysisData, plot_variables, plt):
    g=plot_variables(analysisData, ['PVHC', 'Redes', 'Edu', 'Edad'])
    for ax in g.axes.flatten():
        # rotate x axis labels
        ax.set_xlabel(ax.get_xlabel(), rotation = 90)
        ax.xaxis.set_tick_params(rotation=90)
    plt.gca()
    return ax, g


@app.cell
def __():
    #plot_variables(analysisData, ['PVHC','AB_Oir', 'AB_Hablar', 'AB_Ver', 'AB_Mover', 'AB_Agarrar', 'AB_Aprender', 'AB_ComerVestir', 'AB_Relacionarse', 'AB_CardiacoRespiratorio'])
    return


@app.cell
def __():
    #plot_variables(analysisData,['PVHC','PVC_Cigarrillo', 'PVC_Alcohol', 'PVC_Marihuana', 'PVC_Inhalantes', 'PVC_Cocaina', 'PVC_Basuco', 'PVC_Heroina', 'PVC_Pepas', 'PVC_Otras'])
    return


@app.cell
def __():
    #plot_variables(analysisData,['PVHC','Edu', 'Redes'])
    return


@app.cell
def __():
    #plot_variables(analysisData,['PVHC','Genero', 'Edad', 'Edu', 'Redes'])
    return


@app.cell
def __(analysisData, ax, plot_variables, plt):
    gg = plot_variables(analysisData,['PVHC','MotivoInicio','MotivoPermanencia'])
    for ax2 in gg.axes.flatten():
        # rotate x axis labels
        ax2.set_xlabel(ax.get_xlabel(), rotation = 90)
        ax2.xaxis.set_tick_params(rotation=90)
    plt.gca()
    return ax2, gg


@app.cell
def __():
    #sns.heatmap(pd.crosstab(analysisData['MotivoInicio'],analysisData['MotivoPermanencia'],normalize=True).transpose(),annot=True, fmt=".1f",cmap="crest",cbar=False)
    #plt.subplots_adjust(left=0.5,bottom=0.5)
    #plt.gca()
    return


@app.cell
def __(analysisData, gg, sns):
    z = analysisData.copy()

    sns.boxplot(x='Edu',y='AHC', data=analysisData)
    for ax in gg.axes.flatten():

       ax.set_xlabel(ax.get_xlabel(), rotation = 90)
    return ax, z


@app.cell
def __():
    #plot_variables(analysisData,['PVHC', 'Edu', 'Genero', 'MayorContacto', 'AB_Oir', 'AB_Hablar', 'AB_Ver', 'AB_Mover', 'AB_Agarrar', 'AB_Aprender', 'AB_ComerVestir', 'AB_Relacionarse', 'AB_CardiacoRespiratorio', 'PVC_Cigarrillo', 'PVC_Alcohol', 'PVC_Marihuana', 'PVC_Inhalantes', 'PVC_Cocaina', 'PVC_Basuco', 'PVC_Heroina', 'PVC_Pepas', 'PVC_Otras'])
    return


@app.cell
def __():
    #g = plot_variables(analysisData,['PVHC', 'Edad', 'Edu', 'MayorContacto', 'PVC_Alta', 'PVC_Media', 'PVC_Baja'])
    #for ax in g.axes.flatten():
    #    # rotate x axis labels
    #    ax.set_xlabel(ax.get_xlabel(), rotation = 90)
    #    # rotate y axis labels
    #    ax.set_ylabel(ax.get_ylabel(), rotation = 0)
    #    # set y labels alignment
    #    ax.yaxis.get_label().set_horizontalalignment('right')
    #g
    return


@app.cell
def __():
    #gg=plot_variables(analysisData, ['PVHC','Edu'])
    #for ax in gg.axes.flatten():
    #    # rotate x axis labels
    #    ax.set_xlabel(ax.get_xlabel(), rotation = 90)
    return


if __name__ == "__main__":
    app.run()
