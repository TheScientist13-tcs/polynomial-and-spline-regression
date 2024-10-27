import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from extended_linear_models import NaturalCubicSplines
from extended_linear_models import PolynomialRegression
import plotly.express as px

np.random.seed(1111)


def generate_data(samples=300):
    num_samples = 300
    f = lambda x: np.sin(2 * x) + np.log(10 * x)
    X = np.random.uniform(0, 5, size=num_samples)
    Y_true = np.array(list(map(f, X)))
    noise = np.random.normal(0, 1, size=num_samples)
    Y_obs = Y_true + noise
    return (X, Y_obs, Y_true)


def main():
    st.set_page_config(layout="wide")
    st.markdown("# Polynomial and Spline Regression Demo")
    st.markdown(
        "##### By: Dharyll Prince M. Abellana | Assistant Professor of Computer Science | University of the Philippines Cebu",
    )
    X, Y_obs, Y_true = generate_data()

    ### sidebar/configure knot locations
    with st.sidebar:
        hide_scatter = st.checkbox("Hide Scatter?")
        hide_true_func = st.checkbox("Hide True Function?")
        hide_poly = st.checkbox("Hide Polynomial Regression?")
        hide_spline = st.checkbox("Hide Spline Regression?")
        ## Fit spline

        if not hide_spline:
            st.header("Set Spline Knots")
            knot_1 = st.slider(
                r"""Set Knot 1 ($\xi_1$)""", min_value=0.0, max_value=np.max(X)
            )
            knot_2 = st.slider(
                r"""Set Knot 2 ($\xi_2$)""", min_value=0.0, max_value=np.max(X)
            )
            knot_locs = [knot_1, knot_2]
            cub_reg = NaturalCubicSplines(knot_locs).fit(X, Y_obs)
            y_pred_spline = cub_reg.predict(X)

        ## Fit Polynomial Regression
        if not hide_poly:
            st.header("Set Polynomial Degree")
            p = st.number_input(
                "Set polynomial degree", min_value=1, max_value=100, step=1
            )
            poly_reg = PolynomialRegression(degree=p).fit(x=X, y=Y_obs)
            poly_pred = poly_reg.predict(X)

    ## Plot

    #### Create the figure
    fig = go.Figure()

    #### Add a scatterplot
    if not hide_scatter:
        fig.add_trace(
            go.Scatter(
                x=X,
                y=Y_obs,
                mode="markers",
                name="Observations",
                line=dict(color="#D2D3C9"),
            )
        )
    #### Add true function
    if not hide_true_func:
        sorted_pairs = sorted(zip(X, Y_true))
        sorted_x, sorted_y_true = zip(*sorted_pairs)

        fig.add_trace(
            go.Scatter(
                x=sorted_x,
                y=sorted_y_true,
                mode="lines",
                name="True",
                line_dash="dash",
                line_width=3,
                line=dict(color="#0E918C"),
            )
        )

    #### Add spline prediction
    if not hide_spline:
        sorted_pairs = sorted(zip(X, y_pred_spline))
        sorted_x, sorted_y_spline = zip(*sorted_pairs)

        fig.add_trace(
            go.Scatter(
                x=sorted_x,
                y=sorted_y_spline,
                mode="lines",
                name="Spline",
                line=dict(color="#BB2205"),
            )
        )

        fig.add_vline(
            x=knot_1,
            line_color="#A04747",
            line_dash="dash",
            name="knot 1",
            annotation_text="Knot 1",
            annotation_position="top right",
        )

        fig.add_vline(
            x=knot_2,
            line_color="#A04747",
            line_dash="dash",
            name="knot 2",
            annotation_text="Knot 2",
            annotation_position="top right",
        )

    ### Add Polynomial Regression
    if not hide_poly:
        sorted_pairs = sorted(zip(X, poly_pred))
        sorted_x, sorted_y_poly = zip(*sorted_pairs)

        fig.add_trace(
            go.Scatter(
                x=sorted_x,
                y=sorted_y_poly,
                name=f"Polynomial (p={p})",
                line=dict(color="#F6830F"),
            )
        )

    #### Display plot
    fig.update_layout(width=2000, height=500, xaxis_title="X", yaxis_title="Y")
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
