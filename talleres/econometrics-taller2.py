import marimo

__generated_with = "0.1.88"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Método hedónico para precio de vivienda
        #### Santiago Alonso Perez Rubiano - Código: 200822341
        Utilizando datos proporcionados sobre características y precio de varias viviendas, realizaremos un análisis para encontrar relaciones lineales que ayuden a explicar el ¿cómo se determina el valor de una vivienda?
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""
    ## Análisis de variables
    Para empezar nuestro análisis de la información proporcionada ($n=321$) calcularemos estadísticos descriptivos básicos (comando `desc` en Stata) y graficaremos histogramas de cada variable (comando `histogram` de Stata):
    """)
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

    data = pd.read_stata('/mnt/HyperV/HPRICE3.DTA')
    data = data[['price','age','area','rooms','baths','inst']]
    data.describe()
    return corrplot, data, mo, np, pd, plt, sns, stats


@app.cell
def __(data, plt):
    data.hist(figsize=(15, 10))
    plt.gca()
    return


@app.cell
def __(mo):
    mo.md(r"""
    Algunas observaciones que podemos hacer a primera vista son:

    - Las variables **price** y **age** son positivas y tienen una desviación estándar mayor a su correspondiente media, luego se nota visualmente que ambas variables están altamente concentrada en un rango reducido de valores.
    -  Hay unos picos de datos evidentes en las variables **rooms** y **baths**, y un claro declive en los valores máximos.
    -  Hay unos picos de datos evidentes en la variable **inst** en los múltiplos de 5 mil pies (aprox 1 milla) o de 7 mil pies.

    Estas dos observaciones nos sugieren lo siguiente:

    - Las variable **price** y **age** parece tener una distribución log-normal
    - La versión categorica de **inst** basada en millas podría ayudar a dividir los datos en grupos más significativos.

    Luego, al analizar las correlaciones entre estas variables:
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

    - Hay una relación lineal positiva de **price** con: **area**, **rooms**, **baths** e **inst**, sin embargo la más fuerte de todas es **area**, la cual a su vez está altamente correlacionada con **baths** y **rooms** (lo que podría generar colinealidades nocivas para una regresión múltiple).
     - Hay una relación lineal negativa entre **age** y **price** (la cual podría ser más evidente en la versión logarítmica de ambas variables). La variable **age** no tiene correalciones importantes con **area**, **baths** o **rooms**, pero si con **inst**
     - La correlación entre **inst** y **price** parece menos fuerte que la correlación entre **inst** y **age** (por lo que podrían introducirse colinealidades al considerat tanto **age** como **inst** como variables exógenas a la vez).

    En otras palabras, parece que las variables más importantes a considerar a la hora de evaluar un modelo de regresión lineal múltiple serían **area** y **age**, pues el resto de variables parece que pueden ser explicadas en función de estas dos.

    ## Regresión lineal múltiple

    Recordaondo que de acuerdo a [Ben J. Sopranzetti](https://www.researchgate.net/profile/Ben-Sopranzetti/publication/283842709_Hedonic_Regression_Models/links/5703a02608ae646a9da9b573/Hedonic-Regression-Models.pdf):

    "En su forma más básica, los modelos de precios hedónicos descompononen el precio de un bien en el precio de los bienes individuales que lo consitutyen [...] para examinar como cada componente individual aporta al valor general del bien"


    Y utilizando el modelo hedónico propuesto para la vivienda por el mismo autor:

    \[
        Price = \beta_0 + \beta_1S +\beta_2N + \beta_3L + \beta_4C +\beta_5T
    \]

    Donde:

     - S representa característica sestructurales
     - N representa características del barrio
     - L representa la localización dentro de un mercado dado
     - C representa las condiciones del contrato de venta (conjunto cerrado, individual, etc...)
     - T representa la fecha y hora en la que la transacción de venta fue observada

    Podemos proponer un modelo cercano que utiliza las variables que tenemos a nuestra disposición así:

    \[
        Price_i = \beta_0 + \beta_1Age_i + \beta_2Area_i + \beta_3Rooms_i + \beta_4Baths_i + \beta_5Inst_i + \mu_i
    \]

    Para realizar un análisis de regresión lineal (aunque podríamos relizar análisis log-lineales, box-cox, etc.., pero queda por fuera del alcance de este ejercicio):

    """)
    return


@app.cell
def __(data, mo):
    import statsmodels.formula.api as smf

    mod = smf.ols('price ~ age + area + rooms + baths + inst', data=data)
    res = mod.fit()
    mo.md(f"""
    {mo.as_html(res.summary())}
    ## Intepretación
    """)
    return mod, res, smf


@app.cell
def __(mo):
    mo.md(r"""

    De acuerdo con los resultados del método de mínimos cuadrados ordinarios podemos concluir que:

     - (**2.b.** Significancia) Ya que la prueba del estadístico F permite rechazar la hipótesis nula ($H_0: \beta_i=0$) con una alto nivel de confianza ($>99\%$) entonces podemos decir con significancia estadística que almenos una de las variables independientes consideradas tiene una relación lineal con el **precio**.
      - (**2.c.** Depedencia) Dado el coeficiente de determinación (Adj. R-squared: 0.53) hallado podemos decir que un poco más de la mitad de la variabilidad del precio se puede explicar por la variabilidad de las variables independientes consideradas.

    Es decir que a nivel global podemos decir con significancia estadística que estamos considerando factores que ayudan a explicar en buena medida el precio de las viviendas de forma lineal.

    A nivel particular podemos concluir lo siguiente acerca de cada uno de los coeficientes hallados (**2.d**, **2.e**, **2.f**):


    | Variable   | Interpretación | Significancia | Confianza |
    | :-------- | :------- | :-------- | :------- |
    | $\beta1$ (age) | (Inversamente proporcional) Un incremento en un año de edad **disminuye** el precio de la vivienda en 351 dólares en promedio | La prueba t indica que el resultado es estadísticamente significativo con una confianza de más del **99%** pues el p-value es menor a 0.01 | Con un 95% de confianza podemos decir que el coeficiente es negativo (el intervalo no cruza el eje) y tiene un efecto en el orden de los cientos de dólares  |
    | $\beta_2$ (area) | (Directamente proporcional) Un incremento en un pie cuadrado de área **aumenta** el precio de la vivienda en 29 dólares en promedio | La prueba t indica que el resultado es estadísticamente significativo con una confianza de más del **99%** pues el p-value es menor a 0.01 | Con un 95% de confianza podemos decir que el coeficiente es positivo (el intervalo no cruza el eje) y tiene un efecto en el orden de las decenas de dólares  |
    | $\beta_3$ (rooms) | (Directamente proporcional) Un incremento en un cuarto **aumenta** el precio de la vivienda en 3 815 dólares en promedio | La prueba t indica que el resultado **NO** es estadísticamente significativo pues el p-value está por encima del 0.1 | Dado que el invervalo cruza el origen, no podemos decir con mucha confianza nada acerca de esta variable |
    | $\beta_4$ (baths) | (Directamente proporcional) Un incremento en un cuarto de baño **aumenta** el precio de la vivienda en 11 130 dólares en promedio | La prueba t indica que el resultado es estadísticamente significativo | Dado que el invervalo cruza el origen, no podemos decir con mucha confianza nada acerca de esta variable |
    | $\beta_5$ (inst) |  (Inversamente proporcional) Un incremento en la distancia en pies a la interestatal **disminuye** el precio de la vivienda en 36 centavos en promedio  | La prueba t indica que el resultado es estadísticamente levemente significativo pues el p-value está apenas por encima del 0.1 | Dado que el invervalo cruza el origen, no podemos decir con mucha confianza nada acerca de esta variable |
    | $\beta_0$ (Intercept) | Dado que en nuestro modelo no tiene sentido estimar el precio de una vivienda nueva con cero área y ningún baño o cuarto, entonces no podemos dar una interpretación clara a este valor. Sin embargo podríamos intentar explicarlo como la necesidad de tener almenos un baño en la casa para que tenga valor. | La prueba t indica que el resultado **NO** es estadísticamente significativo pues el p-value está por encima del 0.1 | Dado que el invervalo cruza el origen, no podemos decir con mucha confianza nada acerca de esta variable |

    """)
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
