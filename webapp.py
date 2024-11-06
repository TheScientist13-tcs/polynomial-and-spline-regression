import streamlit as st
import numpy as np
import plotly.graph_objects as go
from extended_linear_models import NaturalCubicSplines
from extended_linear_models import PolynomialRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(1111)


def generate_data(num_samples=300):
    f = lambda x: np.sin(2 * x) + np.log(10 * x)
    X = np.random.uniform(0, 5, size=num_samples)
    Y_true = np.array(list(map(f, X)))
    noise = np.random.normal(0, 1, size=num_samples)
    Y_obs = Y_true + noise
    return (X, Y_obs, Y_true)


def main():
    st.set_page_config(page_title="Polynomial and Splines Regression", layout="wide")
    st.title("Polynomial and Spline Regression Demo")
    st.markdown(
        "##### By: Dharyll Prince M. Abellana | Assistant Professor of Computer Science | University of the Philippines Cebu",
    )
    X, Y_obs, Y_true = generate_data()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_obs, test_size=0.30, shuffle=True
    )
    ### sidebar/configure knot locations
    with st.sidebar:
        hide_scatter = st.checkbox("Hide Scatter")
        hide_true_func = st.checkbox("Hide True Function")
        hide_poly = st.checkbox("Hide Polynomial Regression")
        hide_spline = st.checkbox("Hide Spline Regression")
        ## Fit spline

        if not hide_spline:
            st.header("Spline Regression")
            auto_select_knot = st.checkbox("Auto configure knots")
            if not auto_select_knot:
                st.markdown("### Set Spline Knots")
                knot_1 = st.slider(
                    r"""Set Knot 1 ($\xi_1$)""",
                    min_value=0.0,
                    max_value=np.max(X),
                    value=0.1,
                )
                knot_2 = st.slider(
                    r"""Set Knot 2 ($\xi_2$)""",
                    min_value=0.0,
                    max_value=np.max(X),
                    value=1.0,
                )
                knot_locs = [knot_1, knot_2]
            else:
                num_knots = st.number_input(
                    "Enter Number of Knots", min_value=2, max_value=100
                )
                num_regions = num_knots + 1
                x_min = np.min(X)
                x_max = np.max(X)
                k_k = lambda k: x_min + k * ((x_max - x_min) / (num_regions))
                knot_locs = [k_k(k) for k in range(1, num_regions)]
            cub_reg = NaturalCubicSplines(knot_locs).fit(X_train, Y_train)
            y_pred_spline = cub_reg.predict(X)
            for_test_mse_y_pred = cub_reg.predict(X_test)
            for_train_mse_y_pred = cub_reg.predict(X_train)
            test_mse_spline = np.round(
                float(mean_squared_error(y_true=Y_test, y_pred=for_test_mse_y_pred)), 3
            )
            train_mse_spline = np.round(
                float(mean_squared_error(y_true=Y_train, y_pred=for_train_mse_y_pred)),
                3,
            )

        ## Fit Polynomial Regression
        if not hide_poly:
            st.header("Set Polynomial Degree")
            p = st.number_input(
                "Set polynomial degree", min_value=1, max_value=100, step=1
            )
            poly_reg = PolynomialRegression(degree=p).fit(x=X_train, y=Y_train)
            poly_pred = poly_reg.predict(X)
            for_test_mse_y_pred_poly = poly_reg.predict(X_test)
            for_train_mse_y_pred_poly = poly_reg.predict(X_train)
            test_mse_poly = np.round(
                float(
                    mean_squared_error(y_true=Y_test, y_pred=for_test_mse_y_pred_poly)
                ),
                3,
            )
            train_mse_poly = np.round(
                float(
                    mean_squared_error(y_true=Y_train, y_pred=for_train_mse_y_pred_poly)
                ),
                3,
            )

    ## Plot

    #### Create the figure
    fig = go.Figure()

    #### Add a scatterplot
    if not hide_scatter:
        with st.sidebar:
            hide_train = st.checkbox("Hide Training Data")
            hide_test = st.checkbox("Hide Testing Data")
        if not hide_train:
            fig.add_trace(
                go.Scatter(
                    x=X_train,
                    y=Y_train,
                    mode="markers",
                    marker=dict(opacity=0.7),
                    name="Train Set",
                    line=dict(color="#D2D3C9"),
                )
            )
        if not hide_test:
            fig.add_trace(
                go.Scatter(
                    x=X_test,
                    y=Y_test,
                    mode="markers",
                    marker=dict(opacity=0.7),
                    name="Test Set",
                    line=dict(color="#A47786"),
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
                name=f"Spline (test mse={test_mse_spline}, train mse={train_mse_spline})",
                line=dict(color="#BB2205"),
            )
        )
        with st.sidebar:
            hide_knot = st.checkbox("Hide Knot Location")
        if not hide_knot:
            for i in range(len(knot_locs)):
                fig.add_vline(
                    x=knot_locs[i],
                    line_color="#A04747",
                    line_dash="dash",
                    name=f"knot {i+1}",
                    annotation_text=f"Knot {i+1}",
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
                name=f"Polynomial (p={p}, test mse={test_mse_poly}, train mse={train_mse_poly})",
                line=dict(color="#F6830F"),
            )
        )

    #### Display plot
    fig.update_layout(width=2000, height=500, xaxis_title="X", yaxis_title="Y")
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
