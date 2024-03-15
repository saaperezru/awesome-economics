import marimo

__generated_with = "0.2.8"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    p = [250,260,270,280,290,300,310,320,330,340,350]
    o = [0,40,80,120,160,200,240,280,320,360,400]
    d = [240,220,200,180,160,140,120,100,80,60,40]
    return d, mo, np, o, p, patches, plt


@app.cell
def __(mo):
    mo.md(
        r"""
        # Primer ejercicio
        ## 1.a. Funciones de oferta y demanda
        Si expresamos las funciones de oferta $Q^o$ y demanda $Q^d$ de forma lineal tendremos:

        \[
        Q^o(p) = m_op + c_o
        \]

        \[
        Q^d(p) = m_dp + c_d
        \]


        Encontremos la pendiente de la curva usando los datos disponibles:

        \[
        m_o = \frac{40-0}{260-250} = \frac{40}{10} = 4
        \]

        \[
        m_d = \frac{220-240}{260-250} = \frac{-20}{10} = -2
        \]

        Luego podemos encontrar las constantes $c_o$ y $c_d$, reemplazando las pendientes y despejando para un punto particular:

        \[
        c_o = Q^o(p) - m_op \Rightarrow c_o = 0 - m_o*250 = - 1000
        \]

        \[
        c_d = Q^d(p) - m_bp \Rightarrow c_d = 240 - m_d*250 = 240 + 500 = 740
        \]

        Con lo que obtenemos las siguientes funciones:

        \[
        Q^o(p) = m_op + c_o = 4p - 1000
        \]

        \[
        Q^d(p) = m_dp + c_d = 740-2p
        \]

        ## 1.b. Equilibrio
        Dadas estas funciones podemos calcular el equilibrio en 

        \[
        \begin{equation}
        \begin{split}
        Q^o(p) & = Q^d(p) \\
        4p - 1000  & = 740-2p \\
        6p  & = 1740 \\
        p^*  & = 290
        \end{split}
        \end{equation}
        \]

        Lo que quiere decir que la cantidad de equilibrio será:

        \[
        \begin{equation}
        \begin{split}
        Q^o(p^*) & = 4p* - 1000 \\
        & = 4*290 - 1000 \\
        q^*  & = 160
        \end{split}
        \end{equation}
        \]
        """
    )
    return


@app.cell
def __(d, np, o, p, plt):
    fig, ax = plt.subplots(1,1)
    ax.plot(o,p, linestyle='--', marker='o', color='r')
    ax.plot(d,p, linestyle='--', marker='o', color='b')

    ax.set_xlim((0,250))
    ax.set_yticks(np.append(ax.get_yticks(),[290]))
    ax.set_ylim(ax.get_ylim())
    ax.set_xticks(np.append(ax.get_xticks(),[160]))
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Price')
    ax.annotate('q*',xy=(162,242), xytext=(162,242),annotation_clip=False, color='blue')
    ax.annotate('p*',xy=(2,292), xytext=(2,292),annotation_clip=False, color='blue')

    ax.axvline(160, linestyle='dotted')
    ax.axhline(290, linestyle='dotted')
    return ax, fig


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 1.c. Elasticidad
        Para encontrar la elasticidad debemos aplicar la fórmula de la elasticidad para funciones lineales:

        \[
        \epsilon(p) = \frac{\frac{\Delta q}{q}}{\frac{\Delta p}{p}} = \frac{p}{q}\frac{\Delta q}{\Delta p} = p\frac{Q'(p)}{Q(p)} = p\frac{m}{mp+c}
        \]

        Lo que resulta en las siguientes elasticidades de precio para la oferta ($\epsilon_o$) y la demanda ($\epsilon_d$):

        \[
        \epsilon_o(p) = p\frac{m_o}{m_op+c_o} = \frac{4p}{4p - 1000}
        \]

        \[
        \epsilon_d(p) = p\frac{m_d}{m_dp+c_d} = \frac{-2p}{740-2p}
        \]

        Lo que quiere decir que en el punto de equilibrio tenemos las siguientes elasticidades:

        \[
        \epsilon_o(290) = \frac{4*290}{4*290-1000} = \frac{1160}{160} = 7.25
        \]

        \[
        \epsilon_d(290) = \frac{-2*290}{740-2*290} = \frac{-580}{1320} = -0.43
        \]

        ## 1.d. Subsidios

        Tras aplicar un subsidio pagado a los oferentes de 30 mil la nueva función de oferta sería:

        \[
        \begin{equation}
        \begin{split}
        Q^{o}_s(p) & = m_o(p+30) + c_o \\
        & = m_op +30m_o +c_o \\
        & = 4p + 120 -1000 \\
        & = 4p - 880 \\
        \end{split}
        \end{equation}
        \]

        ### 1.d.ii. Equilibrio

        Con lo que el nuevo equilibrio quedaría en:

        \[
        \begin{equation}
        \begin{split}
        Q^o_s(p) & = Q^d(p) \\
        4p - 880 & = 740-2p \\
        6p & = 1620 \\
        p^{**} & = 270
        \end{split}
        \end{equation}
        \]

        Luego, la cantidad de equilibrio será:

        \[
        \begin{equation}
        \begin{split}
        Q^o_s(270) & = 4*270 - 880 \\
        q^{**} & = 200
        \end{split}
        \end{equation}
        \]

        Gráficamente marcaremos el antigüo equilibrio marcado con $p^{*}$ y el nuevo equilibrio con $p^{**}$:

        """
    )
    return


@app.cell
def __(d, np, o, p, plt):
    def draw_subsidy(ax):
        ax.plot(o,p, linestyle=(0, (1, 10)), color='r')

        ax.plot(o,np.array(p)-30, linestyle='--', marker='o', color='r')
        ax.axvline(160, linestyle=(0, (1, 10)))
        ax.axhline(290, linestyle=(0, (1, 10)))

        ax.plot(d,p, linestyle='--', marker='o', color='b')

        ax.set_xlim([120,250])
        ax.set_xticks(np.append(ax.get_xticks(),[160,120,240]))
        ax.set_ylim([250,310])
        ax.set_yticks(np.append(ax.get_yticks(),[290,310]))
        ax.set_xlabel('Quantity')
        ax.set_ylabel('Price')


        ax.axvline(200, linestyle='dotted')
        ax.axhline(260, linestyle='dotted')
        ax.axhline(270, linestyle='dotted')

        ax.axhline(300, linestyle='dotted')
        ax.annotate('q**',xy=(202,252), xytext=(202,252),annotation_clip=False, color='blue')
        ax.annotate('p*',xy=(122,292), xytext=(122,292),annotation_clip=False, color='blue')
        ax.annotate('p**',xy=(122,272), xytext=(122,272),annotation_clip=False, color='blue')
        ax.annotate('pq*',xy=(122,262), xytext=(122,262),annotation_clip=False, color='blue')
        ax.annotate('q*',xy=(162,252), xytext=(162,252),annotation_clip=False, color='blue')

    _, ax2 = plt.subplots(1,1)
    draw_subsidy(ax2)
    ax2
    return ax2, draw_subsidy


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 1.d.i. Costo para el estado
        El gobierno pagará 30 mil por cada seguro vendido en el nuevo equilibrio, es decir que el costo será:

        \[
        \begin{equation}
        \begin{split}
        Costo_e &= q^{**} * p_s \\
        & = 200 * 30.000 \\
        & = 6.000.000
        \end{split}
        \end{equation}
        \]

        Lo que se puede visualizar en la gráfica de oferta y demanda así:
        """
    )
    return


@app.cell
def __(draw_subsidy, patches, plt):
    _, ax3 = plt.subplots(1,1)
    draw_subsidy(ax3)
    rect = patches.Rectangle((0,270), 200, 30, linewidth=1, fill=False, edgecolor='g', hatch='//')
    ax3.add_patch(rect)
    ax3
    return ax3, rect


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 1.d.iii. Beneficio para los consumidores

        Los consumidores se verán beneficiados en la medida que ahorran (dejan de pagar) un excedente superior al que existía en el precio de equilibrio original ($p^*$). Es decir, calculamos el excedente del nuevo precio de equilibrio ($p^{**}$) y le restamos el excedente en el precio de equilibrio original ($p^*$), lo que nos indica el valor que ahorran los consumidores gracias a la existencia del subsidio.

        Esta diferencia de ahorro se puede calcular con la integral definida:

        \[
        \int_{p^{**}}^{p^*} Q^d(p)dp
        \]

        Que se vería así en la gráfica de oferta-demanda:

    """)
    return


@app.cell
def __(draw_subsidy, patches, plt):
    _, ax4 = plt.subplots(1,1)
    draw_subsidy(ax4)
    ax4.add_patch(patches.Rectangle((0,270), 160, 20, linewidth=0, fill=False, edgecolor='b', hatch='o'))
    ax4.add_patch(patches.Polygon([(160,270),(200,270),(160,290)], linewidth=0, fill=False, edgecolor='b', hatch='o'))
    ax4
    return ax4,


@app.cell
def __(mo):
    mo.md(r"""Si usamos la integral indefinida de $Q^d$ que llamaremos $E^d$:

        \[
        \begin{equation}
        \begin{split}
        E^d(p) &= \int Q^d(p)dp\\
        & =  \int (m_dp + c_d)dp\\
        & = \frac{m_dp^2}{2} + c_dp \\
        & = 740p -  p^2
        \end{split}
        \end{equation}
        \] 

        Podemos entonces calcular el excedente que beneficia a los consumidores gracias al subsidio así:

        \[
        \begin{equation}
        \begin{split}
        \int_{p^{**}}^{p^*} Q^d(p)dp  &= E^d(p^*) - E^d(p^{**}) \\
        & = 740p^* -{p^*}^2 - 740p^{**} + {p^{**}}^2 \\
        & = 740*290 - 290^2 - 740*270 + 270^2 \\
        & = 3600
        \end{split}
        \end{equation}
        \] 

        Lo que también se puede calcular de forma geométrica considerando que el área con burbujas azules en la figura es la suma del rectángulo hasta $q^*$ más el triángulo de ahí hasta $q^{**}$:

        \[
        (p^*-p^{**})q^* + \frac{(p^*-p^{**})(q^{**}-q^{*})}{2} = \\
        (p^*-p^{**}) (q^* + \frac{(q^{**}-q^{*})}{2}) =\\
        (p^*-p^{**}) (q^* + \frac{(q^{**}-q^{*})}{2}) =\\
        (p^*-p^{**}) (\frac{q^{**}+q^{*}}{2}) =\\
        (p^*-p^{**}) \frac{(m_op^{**}+m_op^{*}+2c)}{2} = \\
        (20) \frac{(-2*270-2*290+2*740)}{2} = \\
        (20) * \frac{(360)}{2} = 3600
        \] 

        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 1.d.iv Beneficio para los oferentes

        Dado que el oferente se beneficia del aumento en ventas que causa el subsidio, podemos seguir el mismo razonmiento del punto anterior y calcular el beneficio de los oferentes encontrando el área bajo la curva de la función de oferta $Q^o(p)$ entre el precio del nuevo equilibrio y el precio al que estarían dispuestos a vender los oferentes la cantidad del equilibrio existente antes del subsidio ($p^{q^{**}}$). Gráficamente esto se vería así:
        """
    )
    return


@app.cell
def __(draw_subsidy, patches, plt):
    _, ax5 = plt.subplots(1,1)
    draw_subsidy(ax5)
    ax5.add_patch(patches.Rectangle((0,260), 160, 10, linewidth=0, fill=False, edgecolor='r', hatch='o'))
    ax5.add_patch(patches.Polygon([(160,260),(200,270),(160,270)], linewidth=0, fill=False, edgecolor='r', hatch='o'))
    ax5
    return ax5,


@app.cell
def __(mo):
    mo.md(r"""O lo que es equivalente:""")
    return


@app.cell
def __(draw_subsidy, patches, plt):
    _, ax6 = plt.subplots(1,1)
    draw_subsidy(ax6)
    ax6.add_patch(patches.Rectangle((0,290), 160, 10, linewidth=0, fill=False, edgecolor='r', hatch='o'))
    ax6.add_patch(patches.Polygon([(160,290),(200,300),(160,300)], linewidth=0, fill=False, edgecolor='r', hatch='o'))
    ax6
    return ax6,


@app.cell
def __(mo):
    mo.md(r"""
        Entonces podemos calcular el precio al que estarían dispuestos a vender la cantidad de equilibrio antes del subsidio despejando en la función de oferta:

        \[
        \begin{equation}
        \begin{split}
        Q_s^o(p^{q^{**}}) &= 160\\
        4p^{q^{**}} - 880 & =  160\\
        p^{q^{**}} & = \frac{1040}{4}\\
        & =  260
        \end{split}
        \end{equation}
        \] 

        Luego, si usamos la integral indefinida de $Q^o$ que llamaremos $E^o$:

        \[
        \begin{equation}
        \begin{split}
        E^o(p) &= \int Q^o(p)dp\\
        & =  \int (m_op + c_o)dp\\
        & = \frac{m_op^2}{2} + c_op \\
        & = \frac{4p^2}{2} - 1000p \\
        & = 2p^2 - 1000p
        \end{split}
        \end{equation}
        \] 

        Podemos entonces calcular el excedente que beneficia a los consumidores gracias al subsidio así:

        \[
        \begin{equation}
        \begin{split}
        \int_{p^{**}}^{p^*} Q^o(p)dp  &= E^o(p^*) - E^o(p^{**}) \\
        & = 2{p^*}^2 - 1000p^* - 2{p^{**}}^2 + 1000p^{**} \\
        & = 2*290^2 - 1000*290 - 2*270^2 + 1000*270 \\
        & = 2400
        \end{split}
        \end{equation}
        \] 

        Lo que también se puede calcular de forma geométrica considerando que el área con burbujas azules en la figura es la suma del rectángulo hasta $q^*$ más el triángulo de ahí hasta $q^{**}$:

        \[
        (p^*-p^{**})q^* + \frac{(p^*-p^{**})(q^{**}-q^{*})}{2} = \\
        (p^*-p^{**}) (q^* + \frac{(q^{**}-q^{*})}{2}) =\\
        (p^*-p^{**}) (q^* + \frac{(q^{**}-q^{*})}{2}) =\\
        (p^*-p^{**}) (\frac{q^{**}+q^{*}}{2}) =\\
        (p^*-p^{**}) \frac{(m_op^{**}+m_op^{*}+2c)}{2} = \\
        (20) * \frac{(4*270+4*290-2*1000)}{2} = \\
        (20) * \frac{(240)}{2} = 2400
        \] 

        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### 1.d.v. El caso del SOAT en Colombia

        En nuestro caso anteriormente desarrollado la demanda era menos elástica que la oferta, razón por la cual al cambio de precio la demanda no incremento mucho sus cantidades. Este es un comportamiento similar al del SOAT en Colombia ya que segun el periódico El Tiempo el descuento del 50% del SOAT “lejos de contribuir a reducir la evasión de la póliza, estimular su compra y contribuir de alguna manera a bajar la accidentalidad vial, está dejando un millonario hueco en los recursos del sector de la salud a cargo de la Adres”. Esto nos demuestra que la demanda del soat es inelástica porque como dice el artículo a pesar del descuento en el precio la demanda no se incrementó.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""
    #Segundo ejercicio
    ## 2.a. Equilibrio
    Dadas las funciones de oferta y demanda:

    \begin{equation}
      \begin{split}
        Q_o(p) &= p-100
      \end{split}
    \quad\&\quad
      \begin{split}
        Q_d(p) &= 4400 - 2p
      \end{split}
    \end{equation}

    Encontramos el equilibrio igualando $Q_o = Q_d$:

    \begin{equation}
      \begin{split}
        Q_o(p) &= Q_d(p) \\
        p-100 &= 4400-2p \\
        3p &= 4500 \\
        p* &= 1500
      \end{split}
    \end{equation}

    Lo que quiere decir que la cantidad en equilibrio es:

    \[
    q* = Q_o(1500) = 1400
    \]

    ## 2.b. Fijar precio máximo

    Para entender el impacto de un precio máximo ($1200) podemos:

     1. Comparar este precio con el de equilibrio($1500), para notar que este precio máximo impedirá al mercado quilibrarse en el precio ideal.
     2. Calcular tanto oferta como demanda en el precio exigido y comparar:

    \begin{equation}
      \begin{split}
        Q_o(1200) &\quad Q_d(1200) \\
        1200 - 100 &\quad 4400-2*1200 \\
        1100 &\quad 4500 \\
      \end{split}
    \end{equation}

    Que evidencia una ineficiencia en forma de exceso de **demanda** con respecto a la oferta, que es causada por la imposición de precios en el mercado.

    """)
    return


@app.cell
def __(np):
    def draw_plane_tic(ax):
        p = np.arange(0,2500)
        o = p - 100
        d = 4400-2*p
        ax.plot(o,p, linestyle='--', color='r')
        ax.plot(d,p, linestyle='--', color='b')
        ax.axvline(1400, linestyle=(0, (1, 10)))
        ax.axhline(1500, linestyle=(0, (1, 10)))
        
        ax.set_xlim([0,2000])
        ax.set_xticks(np.append(ax.get_xticks(),[]))
        ax.set_ylim([0,2300])
        ax.set_yticks(np.append(ax.get_yticks(),[]))
        ax.set_xlabel('Quantity')
        ax.set_ylabel('Price')
        ax.annotate('q*',xy=(1400,22), xytext=(1400,22),annotation_clip=False, color='blue')
        ax.annotate('p*',xy=(22,1500), xytext=(22,1500),annotation_clip=False, color='blue')
    return draw_plane_tic,


@app.cell
def __(draw_plane_tic, plt):
    _, ax7= plt.subplots(1,1)
    draw_plane_tic(ax7)
    ax7.axhline(1200)
    ax7
    return ax7,


@app.cell
def __(mo):
    mo.md(r"""

    ## 2.c. Fijar precio mínimo

    Para entender el impacto de un precio máximo ($1800) podemos:

     1. Comparar este precio con el de equilibrio($1800), para notar que este precio máximo impedirá al mercado equilibrarse en el precio ideal.
     2. Calcular tanto oferta como demanda en el precio exigido y comparar:


    \begin{equation}
      \begin{split}
        Q_o(1200) &\quad Q_d(1200) \\
        1800 - 100 &\quad 4400-2*1800 \\
        1700 &\quad 800 \\
      \end{split}
    \end{equation}

    Que evidencia una ineficiencia en forma de exceso de **oferta** con respecto a la demanda, que es causada por la imposición de precios en el mercado

    """)
    return


@app.cell
def __(draw_plane_tic, plt):
    _, ax8 = plt.subplots(1,1)
    draw_plane_tic(ax8)
    ax8.axhline(1800)
    ax8
    return ax8,


@app.cell
def __(mo):
    mo.md(r"""

    ## 2.e. Impuestos

    Al subirse le precio debido al impuesto, los demandantes ya no van a demandar la misma cantidad, ellos demandarían menos cantidad y la curva de la demanda se desplaza hacea la izquierda, resultando en un nuevo punto de equilibrio en P=1300 y Q=1200

    \begin{equation}
      \begin{split}
        Q_o(p) &= Q_d(p+300) \\
        &= 4400 - 2(p+300) \\
        &= 4400 - 2p - 600 \\
        &= 3800 - 2p \\
        Q_o(p)  &= Q_d^t(p) \\
        p - 100  &= 3800 - 2p\\
        3p  &= 3900 \\
        p^{**} &= 1300
      \end{split}
    \end{equation}

    Con lo que obtendremos $q^{**} = Q_d^t(1300) = 3800 - 2*1300 = 1200$

    ## 2.f. Elasticidad

    Utilizando la fórmula de elasticidad para el caso lineal:

    \[
    \epsilon(p) = p\frac{Q'(p)}{Q(p)} = p\frac{m}{mp+c}
    \]

    Sabemos que la elasticidad en el punto de equilibrio para la oferta será:

    \[
    \begin{equation}
      \begin{split}
        \epsilon_o(p) &= p\frac{m_o}{m_op+c_o} \\
        &= \frac{p}{p-100} \\
        \epsilon_o(p*) &= \frac{1500}{1500-100} = 1.071\\
      \end{split}
    \end{equation}
    \]

    Por lo que podemos decir la función de oferta es elastica al precio en el punto de equilibrio, al igual que en cualquier otro punto pues la elasticidad siempre será levemente mayor a 1, es decir, la oferta siempre será un poco elástica.

    Si revisamos la elasticidad de la demanda podemos ver que la oferta es menos elástica que la demanda en el punto de equilibrio
    , por lo que **el impuesto recaerá mayormente sobre las firmas**.


    \[
    \begin{equation}
      \begin{split}
        \epsilon_d(p) &= p\frac{m_d}{m_dp+c_d} \\
        &= \frac{-2p}{4400-2p} \\
        &= -\frac{p}{2200-p} \\
        \epsilon_d(p*) &= -\frac{1500}{2200-1500} = -2.14\\
      \end{split}
    \end{equation}
    \]

    """)
    return


if __name__ == "__main__":
    app.run()
