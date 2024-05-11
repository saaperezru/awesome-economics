import marimo

__generated_with = "0.1.88"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(r"""
    # Estimación de variables categóricas
    #### Santiago Alonso Perez Rubiano - Código: 200822341
        Utilizando la información histórica del incumplimiento crediticio de 1000 préstamos intentaremos estimar la probabilidad de que una solicitud de crédito termine siendo incumplida tras ser aprobada. Para este propósito utilizaremos los métodos de probabilidad lineal, probit y logit e interpretaremos los efectos marginales de cada una de las variables independientes utilizadas.

    ## Análisis de variables
    Para empezar nuestro análisis de la información proporcionada ($n=1000$) calcularemos estadísticos descriptivos básicos (comando `desc` en Stata) y graficaremos histogramas y barras de cada variable (comando `histogram` de Stata):
    """
    )
    return


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from biokit.viz import corrplot
    import seaborn as sns
    from scipy import stats

    import statsmodels.formula.api as smf

    data = pd.read_stata("/mnt/HyperV/Credit_Data.DTA")
    return corrplot, data, mo, np, pd, plt, smf, sns, stats


@app.cell
def __(confidence_ellipse, corrfunc, data, pd, pearsonsig, plt, sns):
    def plot2(x, y, **kws):
        CONTINUOUS_TYPES =  ['int16','int8','float']
        if x.dtype=='category' and y.dtype == 'category':
            sns.heatmap(pd.crosstab(x,y,normalize=True).transpose(),annot=True, fmt=".1f",cmap="crest",cbar=False,**kws)
            #g.set_yticklabels(labels=g.get_yticklabels(), va='center')

    def plot(x, y, **kws):
        CONTINUOUS_TYPES =  ['int16','int8','float']
        if y.dtype=='category' and x.dtype in CONTINUOUS_TYPES:
            sns.boxplot(x=x,y=y,**kws)
        elif x.dtype=='category' and y.dtype in CONTINUOUS_TYPES:
            sns.boxplot(x=x,y=y,**kws)
            #stats.ttest_ind(y, x)
        elif y.dtype in CONTINUOUS_TYPES and x.dtype in CONTINUOUS_TYPES:
            corrfunc(x,y)
            pearsonsig(x,y)
            sns.scatterplot(x=x,y=y,**kws)
            confidence_ellipse(x,y)

    def test(x, **kws):
        if x.dtype=='category':
            sns.barplot(x=x.value_counts().index
                ,y=x.value_counts())
        else:
            sns.histplot(x,kde=True,**kws)

    analysisData = data.copy()
    analysisData['default'] = analysisData['default'].astype('category')
    g = sns.PairGrid(analysisData, vars=data.columns)
    g.map_diag(test)
    g.map_lower(plot)
    g.map_lower(plot2)
    plt.subplots_adjust(left=0.2,bottom=0.2)
    plt.gca()
    return analysisData, g, plot, plot2, test


@app.cell
def __(mo):
    mo.md(r"""
    Las gráficas contienen:

    - La correlacion de Pearson como "PC", seguida de * en aquellos casos en que la depedencia tenga significancia estadistica ($p-value < 0.01$), es decir, que se rechaze la hipotesis nula de $\rho = corr(x,y) = 0$.
    - La [correlacion de la distancia de Székely](https://es.wikipedia.org/wiki/Correlaci%C3%B3n_de_la_distancia) como "DC", que es una [buena medida](https://arxiv.org/pdf/1401.7645.pdf) para identificar relaciones no lineales
    - El "scatterplot" de cada par de variables junto con una [elipse de confianza](https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html) que permite ilustrar la pendiente de la correlación lineal
     - El "boxplot" de cada par de variables continuas y categóricas.
     - La "tabla de contingencia" (con porcentajes, no cantidades) ilustrada como un mapa de calor para cada par de variables categóricas.


    Algunas observaciones que podemos hacer a primera vista son:

    - Se nota un imbalance en los datos, pues hay más información sobre créditos que no se incumplieron, es decir que los modelos construidos con estos datos pueden tener sesgo.
     - En todas las variables categóricas hay categorías con muchos más datos que el resto, por lo que es probable que existan formas de agrupar los datos que hagan las variables más poderosas a la hora de discernir en la probabilidad de incumplir el crédito, por ejemplo, podríamos agrupar todos los propósitos de crédito en 2 o 3 categorías principales que no tengan mucha diferencia entre medias de $default$.
     - Las variables de $age$ y $credit_amount$ tienen una distribución concentrada a la izquierda, lo que quiere decir que puede ser una variable con efectos logarítmicos.

    Luego, al analizar las correlaciones entre estas variables vemos que:

     - De las variables continuas la más relevante parece ser la duración del crédito en meses.
     - De las variables categóricas la más relevante parecen ser las de $housing$, $credit_history$ (estadísticamente significativa además), en línea con lo que se observa empíricamente en los modelos del mundo real.
     - Entre las variables independientes hay relaciones lineales relevantes entre:
        - $duration_in_month$ y $credit_amount$
        - $age$ y $housing$
        - $age$ y $savings$
        - $age$ y $credit_history$
        - $housing$ y $credit_history$
     """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    #Modelado
    Todos los modelos que usaremos asumen la siguiente fórmula:

    \[
         Y = \beta_0 + \beta_1 DurationInMonth + \beta_2 CreditAmount + \beta_3 Age \\
        + \sum^4_{i=1}\beta^c_{i} CreditHistory_i + \sum^9_{i=1}\beta^p_{i} Purpose_i \\
        + \sum^4_{i=1}\beta^s_{i} Savings_i + \sum^3_{i=1}\beta^{pss}_{i} PersonalStatusSex_i  \\
        + \sum^4_{i=1}\beta^s_{i} Housing_i
    \]

    Pues usamos "variables dummy" para representar cada uno de los posibles valores de las variables independientes categóricas nominales, y por lo tanto (y para evitar la ["trampa de la variable dummy"](https://www.learndatasci.com/glossary/dummy-variable-trap/)) cada una de ellas tendrá una categoría de referencia definidas así:

     - $CreditHistory$: *critical account/ other credits existing (not at this bank)*
     - $Purpose$: *(vacation - does not exist?)*
     - $Savings$: *unknown/ no savings account*
     - $PersonalStatusSex$: *female : divorced/separated/married*
     - $Housing$: *for free*

    Usaremos todas las variables disponibles a pesar de que en la práctica solo unas pocas de estas son realmente relevantes en modelos econométricos reales (e.g. [FICO scoring](https://www.investopedia.com/which-fico-scores-do-lenders-use-5116055) no utiliza variables como el [sexo/genero](https://www.federalreserve.gov/econres/notes/feds-notes/gender-related-differences-in-credit-use-and-credit-scores-20180622.html))

    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ##Modelo de Probabilidad Lineal
    Dado que este es un modelo puramente lineal, no hay necesidad de calcular los efectos marginales con ayuda de métodos numéricos pues las derivadas son constantes. Las interpretaciones de los efectos marginales son las usuales (dependiendo del signo y la magnitud):

     - Para variables continuas: El valor asignado a cada coeficiente multiplicado por 100 determina los puntos porcentuales que contribuye la correspondiente variable a la probabilidad de incumplir con el pago del crédito. Por ejemplo, cualquier aumento en el periodo de pago del crédito aumentará la probabilidad de incumplir con el crédito en 0.7 puntos porcentuales.
     - Para variables categóricas: El valor asignado a cada coeficiente multiplicado por 100 determina los puntos porcentuales que la correspondiente CATEGORÍA aumenta la probabilidad de incumplir *con respecto* a la categoría de referencia. Por ejemplo en el caso de $PersonalStatusSex$:
        - El ser hombres divorciado/separados **aumenta** en 6 puntos porcentuales la probabilidad de incumplir el crédito, **con respecto a las mujeres**.
        - El ser hombres viudo **disminuye** en 3 puntos porcentuales la probabilidad de incumplir el crédito, **con respecto a las mujeres**.
        - El ser hombres soltero **disminuye** en 6 puntos porcentuales la probabilidad de incumplir el crédito, **con respecto a las mujeres**.
     - Finalmente, el coeficiente $\beta_0$ (es decir, el intercepto) representa el efecto conjunto que tienen: Las personas que (1) tienen deudas en otros bancos, (2) piden el crédito con propósitos desconocidos, (3) no tienen cuentas de ahorros (o no las tienen reportadas), (4) son mujeres y además no deben pagar alojamiento (a pesar de no ser propietarios).
     - Nótese que algunos coeficientes no son estadísticamente significativos.


    """)
    return


@app.cell
def __(data, mo, smf):
    formula = "default ~ duration_in_month+ credit_amount + age + C(credit_history, Treatment('critical account/ other credits existing (not at this bank)')) + C(purpose, Treatment('(vacation - does not exist?)')) + C(savings, Treatment('unknown/ no savings account')) + C(personal_status_sex,Treatment('female : divorced/separated/married')) + C(housing, Treatment('for free')) +1"
    mod = smf.ols(formula, data=data)
    res = mod.fit()
    mo.md(f"""
    {mo.as_html(res.summary())}
    """)
    return formula, mod, res


@app.cell
def __(mo):
    mo.md(r"""
    ##Logit
    Para este modelo necesitamos calcular el efecto marginal promedio utilizando métodos numéricos pues la derivada de la función $logit$ incluye todos los parámetros del modelo, no solo el de la variable estudiada.

    """)
    return


@app.cell
def __(data, formula, mo, smf):
    modL = smf.logit(formula, data=data)
    resL = modL.fit()
    mo.md(f"""
    {mo.as_html(resL.summary())}
    ### Efectos marginales promedio
    En este caso debemos la interpretación de los coeficientes no es la misma que en el caso de MPL, pero si miramos a los efectos marginales podemos hacer seguir las mismas guías de interpretación: Cambios en la variable correspondiente al efecto marginal tiene un efecto de $100*\beta$ puntos porcentuales en la probabilidad de incumplir.
    {mo.as_html(resL.get_margeff().summary())}
    """)
    return modL, resL


@app.cell
def __(mo):
    mo.md(r"""
    ##Probit
    """)
    return


@app.cell
def __(data, formula, mo, resL, smf):
    modP = smf.probit(formula, data=data)
    resP = modP.fit()
    mo.md(f"""
    {mo.as_html(resP.summary())}
    ### Efectos marginales
    En este caso seguimos la misma interpretación que en el caso de Logit.
    {mo.as_html(resL.get_margeff().summary())}
    """)
    return modP, resP


@app.cell
def __(mo):
    mo.md(r"""
    ## Comparación de efectos marginales en todos los modelos
    Si tabulamos los efectos marginales promedios para Logit y Probit al lado de los coeficientes del modelo MPL podemos ver que los resultados son muy parecidos en TODOS los modelos:
    """)
    return


@app.cell
def __(pd, res, resL, resP):
    pd.set_option("display.max_rows", 30)
    resL.get_margeff().summary_frame()['dy/dx'].to_frame().join(
    resP.get_margeff().summary_frame()['dy/dx'].to_frame(), lsuffix='_L', rsuffix='_P').join(res.params.to_frame(), how='outer')
    #mlpME = res.params.copy()
    return


@app.cell
def __(mo):
    mo.md("""
    ##Curva ROC
    Para evaluar y comparar la efectividad (sensitividad y especificidad) de cada modelo graficamos la curva ROC y calculamos el AUC de cada uno de los modelos estimados:""")
    return


@app.cell
def __(data, plt, res, resL, resP):
    from sklearn import metrics

    fprM, tprM, _ = metrics.roc_curve(data['default'],  res.predict())
    fprP, tprP, _ = metrics.roc_curve(data['default'],  resP.predict())
    fprL, tprL, _ = metrics.roc_curve(data['default'],  resL.predict())


    #create ROC curve
    plt.plot(fprP,tprP,color='red')
    plt.annotate("AUC (Probit) = {:.2f}".format(metrics.auc(fprP, tprP)), xy=(.7, 0.8), color = 'red', fontsize = 10)

    plt.plot(fprL,tprL,color='blue')
    plt.annotate("AUC (Logit) = {:.2f}".format(metrics.auc(fprL, tprL)), xy=(.7, 0.7), color = 'blue', fontsize = 10)

    plt.plot(fprM,tprM,color='orange')
    plt.annotate("AUC (MPL) = {:.2f}".format(metrics.auc(fprM, tprM)), xy=(.7, 0.6), color = 'orange', fontsize = 10)

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.gca()
    return fprL, fprM, fprP, metrics, tprL, tprM, tprP


@app.cell
def __(mo):
    mo.md("""Podemos observar que todos los modelos tienen una área por debajo de la curva ROC muy parecida y por lo tanto podemos decir que su efectividad es también muy parecida.""")
    return


@app.cell
def __(np, plt, stats):
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
        d, p = dist_corr(x,y) 
        #print("{:.4f}".format(d), "{:.4f}".format(p))
        if p > 0.01:
            star=False
            pclr = 'Darkgray'
        else:
            star=True
            pclr= 'Darkblue'
        ax = plt.gca()
        ax.annotate("DC = {:.2f}{}".format(d,'*' if star else ''), xy=(.7, 0.8), xycoords=ax.transAxes, color = pclr, fontsize = 10)

    def pearsonsig(x, y, **kws):
        c, p = stats.pearsonr(x,y)
        #print("{:.4f}".format(c), "{:.4f}".format(p))
        if p > 0.01:
            pclr = 'Darkgray'
            star=False
        else:
            pclr= 'Darkblue'
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
    return (
        confidence_ellipse,
        corrfunc,
        dist_corr,
        hide_current_axis,
        pearsonsig,
    )


if __name__ == "__main__":
    app.run()
