import marimo

__generated_with = "0.1.88"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Correlaciones en la GEIH 2022 DANE

        La [Gran Encuesta Integrada de Hogares GEIH](https://www.dane.gov.co/index.php/estadisticas-por-tema/mercado-laboral) publicada por el DANE (en su verisón mas reciente del [2022](https://microdatos.dane.gov.co/index.php/catalog/771/get-microdata) ) contiene información de ingresos mensuales, edad, educación e intensidad laboral horaria semanal, de residentes Colombianos. A continuación analizaremos estos datos con la ayuda del [modelo econométrico de Mincer](https://es.wikipedia.org/wiki/Funci%C3%B3n_de_ingresos_de_Mincer)
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

    pd.read_stata('/mnt/HyperV/Econometrics-Week3.dta').head()
    return corrplot, mo, np, pd, plt, sns, stats


@app.cell
def __(mo):
    mo.md(r"""
    ## Análisis de variables
    Para empezar nuestro análisis de la información proporcionada ($n=29055$) graficaremos histogramas de cada variable (comando `histogram` de Stata):
    """)
    return


@app.cell
def __(pd, plt):
    data = pd.read_stata('/mnt/HyperV/Econometrics-Week3.dta')
    data = data[['INGLABO','Edad','Horas_Trabajadas','Edu']]
    data.hist(figsize=(15, 10))
    plt.gca()
    return data,


@app.cell
def __(mo):
    mo.md(r"""
    Algunas observaciones que podemos hacer a primera vista son:

    - La variable **INGLABO** esta altamente concentrada en un rango reducido de valores grandes (de 2M a 5M aprox)
    -  Hay unos picos de datos evidentes en la variable **Edu** en los años 6 y 12, lo que concuerda con el tiempo que se requiere para finalizar los periodos de primaria y secundaria respectivamente.
    -  Hay unos picos de datos evidentes en la variable **Horas_Trabajadas** en 48, 30 y 60, que concuerdan con el medio tiempo y el tiempo completo legalmente establecidos en Colombia.

    Estas dos observaciones nos sugieren lo siguiente:

    - La variable **INGLABO** parece tener una distribución log-normal
    - La versión categorica de **Edu** basada en los niveles educativos (primaria, secundaria, universitario, posgrado) podría ayudar a dividir la muestra en subgrupos de estudio
    - La versión categorica de **Horas_Trabajadas** basada en niveles atipicos de trabajo (por debajo o por encima de la jornada laboral legislada) podría ayudar a dividir la muestra en subgrupos de estudio

    ## Análisis de correlaciones

    Ahora, revisemos las relaciones entre todas las posibles parejas de variables en una matriz de correlación (equivalente a usar el comando `pwcorr` en Stata):
    """)
    return


@app.cell
def __(confidence_ellipse, corrfunc, data, pearsonsig, sns):
    g = sns.PairGrid(data)
    g.map_upper(sns.kdeplot)
    g.map_lower(corrfunc)
    g.map_lower(pearsonsig)
    g.map_lower(sns.scatterplot)
    g.map_lower(confidence_ellipse, color='red')
    g.map_diag(sns.histplot, kde=True)
    return g,


@app.cell
def __(mo):
    mo.md(r"""Las gráficas contienen:
     
    - La correlacion de Pearson como "PC", seguida de * en aquellos casos en que la depedencia tenga significancia estadistica ($p-value < 0.01$), es decir, que se rechaze la hipotesis nula de $\rho = corr(x,y) = 0$.
    - La [correlacion de la distancia de Székely](https://es.wikipedia.org/wiki/Correlaci%C3%B3n_de_la_distancia) como "DC", que es una [buena medida](https://arxiv.org/pdf/1401.7645.pdf) para identificar relaciones no lineales
    - El "scatterplot" de cada par de variables junto con una [elipse de confianza](https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html) que permite ilustrar la pendiente de la correlación lineal

    Lo que nos permite realizar las siguientes observaciones:

    - Hay una relacion lineal positiva y estadísticamente significativa entre **Edu** e **INGLABO**
    - Hay una relacion lineal positiva y estadísticamente significativa entre **Horas_Trabajadas** e **INGLABO**
    - Hay una relacion lineal negativa/inversa y estadísticamente significativa entre **Edu** y **Edad**
    - Del resto de parejas se puede decir que: A pesar de qser estadísticamente significativos,  tiene efectos lineales muy bajos por lo que no ayudan mucho a definir un modelo matemático en este momento (nótese que los PC bajos son acompanados de DC mas altos, lo que podría indicar que hay alguna clase de relación que no es evidente al nivel lineal)

    Estudiemos más a fondo estas relaciones graficando con mayor nivel de detalle.

    ### Eduacacion vs Ingreso

    La relacion entre **Edu** e **INGLABO** se ve así:
    """)
    return


@app.cell
def __(data, sns):
    sns.jointplot(data=data,x="Edu", y="INGLABO",kind="reg")
    return


@app.cell
def __(mo):
    mo.md(r"""Lo que concuerda con el modelo econométrico escogido (Mincer), pues hay una correlación lineal evidente entre los ingresos y los años de esutdio. Sin embargo, aquí se puede notar que la regresion lineal no logra caputar un aparente incremento en la pendiente en los rangos mas altos de escolaridad (Universitaria y Posgrado).""")
    return


@app.cell
def __(data, pd, sns):
    data['HigherEdu'] = pd.cut(data['Edu'], (0,6,12,17,100),labels=['Primaria','Bachillerato','Universidad','Posgrado'])
    sns.lmplot(data,x="Edu",y="INGLABO", hue="HigherEdu")
    return


@app.cell
def __(mo):
    mo.md(r"""Lo que es mucho más fácil de observar en un gráfico de barras de la versión categórica de los años de educación""")
    return


@app.cell
def __(data, sns):
    sns.barplot(data,x='HigherEdu',y='INGLABO')
    return


@app.cell
def __(mo):
    mo.md(r"""
    ### Eduacación vs Edad

    Ahora exploremos la relacion entre **Edu** y **Edad**:""")
    return


@app.cell
def __(data, sns):
    sns.jointplot(data=data,x="Edu", y="Edad",kind="reg")
    return


@app.cell
def __(mo):
    mo.md(r"""Aquí nuevamente se puede notar que la regresión lineal no logra caputar un aparente incremento en la pendiente en los rangos de edad más altos, que podría indicar diferencias generacionales en los niveles de educación, i.e. que la educación básica presenta mucha desernción académica en generaciones más antiguas, mientras que en las generaciones más recientes se ve menor deserción:
    #### En menores de 40 años
    """)
    return


@app.cell
def __(data, sns):
    sns.jointplot(data=data[data['Edad']<40],x="Edu", y="Edad",kind="reg")
    return


@app.cell
def __(mo):
    mo.md(r"""#### En mayores de 40 años""")
    return


@app.cell
def __(data, sns):
    sns.jointplot(data=data[data['Edad']>40],x="Edu", y="Edad",kind="reg")
    return


@app.cell
def __(mo):
    mo.md(r"""
    Lo que sugiere que para generaciones más recientes (menores de 40) se sigue el curso esperado de: A más años vividos, mayor educación se tiene, y probablemente mayores ingresos en consecuencia (en concordancia con el modelo de Mincer que sugiere que ["log-earnings age profiles diverge with age across schooling levels"](https://www.nber.org/system/files/working_papers/w9732/w9732.pdf)):
    #### En menores de 40 años
    """)
    return


@app.cell
def __(data, sns):
    sns.lmplot(data[data['Edad']<40], x="Edad", y="INGLABO", hue='HigherEdu')
    return


@app.cell
def __(mo):
    mo.md(r"""#### En mayores de 40 años""")
    return


@app.cell
def __(data, sns):
    sns.lmplot(data[data['Edad']>40], x="Edad", y="INGLABO", hue='HigherEdu')
    return


@app.cell
def __(mo):
    mo.md(r"""
    ### Ingreso vs Horas trabajdas
    Finalmente analizemos en mas detalle el rol de **Horas_Trabajadas** frente a **INGLABO**, a pesar de que estas no son variables directamente estudiadas por el modelo econométrico escogido (pues **Horas_Trabajadas** no es lo mismo que experiencia laboral):""")
    return


@app.cell
def __(data, sns):
    sns.lmplot(data, x="Horas_Trabajadas", y="INGLABO")
    return


@app.cell
def __(data, pd, sns):
    data['Jornada'] = pd.cut(data['Horas_Trabajadas'], (0,35,50,100),labels=['Part-time','Full-time','Extra Fulll Time'])
    sns.histplot(data,x='Jornada')
    return


@app.cell
def __(mo):
    mo.md(r"""Algo curioso a notar (siguiendo las ideas del efecto de la escolaridad de los análisis anteriores) es ver como se comporta esta relación cuando los niveles de educacions son más altos:""")
    return


@app.cell
def __(data, sns):
    sns.lmplot(data, x="Horas_Trabajadas", y="INGLABO", hue='HigherEdu')
    return


@app.cell
def __(mo):
    mo.md(r"""Donde se nota que a mayor nivel de educación, mayor nivel de correlación entre **INGLABO** y **Horas_Trabajadas**""")
    return


@app.cell
def __(data, sns):
    sns.barplot(data,x='Jornada',y='INGLABO')
    return


@app.cell
def __(mo):
    mo.md(r"""
    ##Analisis economico de INGLABO
    Hemos visto que la relevancia de muchas correlaciones cambian de acuerdo al grupo de Edad o al grupo de escolaridad que se estudie, sin embargo vamos a reducirnos a estudiar como el modelo econométrico que escogimos se ve reflejado a nivel general.

    Teniendo en cuenta la formula de Mincer:

    \[
     ln W = ln W_0 + \rho S + \beta_1X + \beta_2X^2
    \]

    Donde $W$ son los ingresos (que se podrian expresar en pesos/hora), $W_0$ es el ingreso de una persona sin experiencia, $S$ son los anos de educacion, y $X$ son los anos de experiencia.

    Podemos ahora explorar un modelo matematico inspirado en la formula de Mincer:

     - Asumamos los **INGLABO** como la variable dependiente en escala logaritmica
     - Usemos la variable **Edu** como variable explicativa/independiente, en linea con lo que propone Mincer en su formula (nuestro **Edu** es el $S$ de la formula)
     - Ignoremos la variable **Edad** pues su capacidad explicativa se puede obtener de la variable **Edu** dada la correlacion que tienen, y ademas no es parte del modelo de referencia de Mincer.
      - Ignoremos $W_0$ pues es una constante que no afecta nuestros analisis de correlacion

     Con esto podriamos aplicar el modelo economico a nuestros datos siguiendo la formula:
     
    \[
     ln (INGLABO) = \rho*Edu
    \]
     
     Lo que gráficamente se vería como:""")
    return


@app.cell
def __(data, np, sns):
    data['LogINGLABO'] = np.log(data['INGLABO'])
    sns.jointplot(data=data,x="Edu", y="LogINGLABO",kind="reg")
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
        ax.annotate("PC = {:.2f}{}".format(c,'*' if star else ''), xy=(.7, 0.9), xycoords=ax.transAxes, color = pclr, fontsize = 12)

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
