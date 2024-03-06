import marimo

__generated_with = "0.2.8"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    p = [250,260,270,280,290,300,310,320,330,340,350]
    o = [0,40,80,120,160,200,240,280,320,360,400]
    d = [240,220,200,180,160,140,120,100,80,60,40]
    return d, mo, np, o, p, plt


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
        p  & = 290
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
        p & = 270
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

        Gráficamente marcaremos el antigüo equilibrio marcado con $p^{**}$ y el nuevo equilibrio con $p^*$:
        
        """
    )
    return


@app.cell
def __(d, np, o, p, plt):
    fig2, ax2 = plt.subplots(1,1)

    ax2.plot(o,p, linestyle=(0, (1, 10)), color='r')

    ax2.plot(o,np.array(p)-30, linestyle='--', marker='o', color='r')
    ax2.axvline(160, linestyle=(0, (1, 10)))
    ax2.axhline(290, linestyle=(0, (1, 10)))

    ax2.plot(d,p, linestyle='--', marker='o', color='b')

    ax2.set_xlim([120,250])
    ax2.set_xticks(np.append(ax2.get_xticks(),[160,120,240]))
    ax2.set_ylim([250,310])
    ax2.set_yticks(np.append(ax2.get_yticks(),[290,310]))
    ax2.set_xlabel('Quantity')
    ax2.set_ylabel('Price')


    ax2.axvline(200, linestyle='dotted')
    ax2.axhline(270, linestyle='dotted')
    ax2.axhline(300, linestyle='dotted')
    ax2.annotate('q**',xy=(202,252), xytext=(202,252),annotation_clip=False, color='blue')
    ax2.annotate('p**',xy=(122,272), xytext=(122,272),annotation_clip=False, color='blue')
    ax2.annotate('q*',xy=(162,252), xytext=(162,252),annotation_clip=False, color='blue')

    return ax2, fig2


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
        
        ### 1.d.iii. Beneficio para los consumidores
        ### 1.d.iv Beneficio para los oferentes
        ### 1.d.v. El caso del SOAT en Colombia
        """
    )
    return


@app.cell
def __():
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
        p &= 1500
      \end{split}
    \end{equation}

    ## 2.b. Fijar precio máximo

    Encontramo
    """)
    return


if __name__ == "__main__":
    app.run()
