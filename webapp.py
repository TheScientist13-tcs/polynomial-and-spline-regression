import streamlit as st
import numpy as np
import plotly.graph_objects as go
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from numpy.polynomial import Chebyshev

np.random.seed(100)


def generate_data(num_samples=40):
    f = lambda x: np.sin(2 * x) + np.log(10 * x)
    X = np.random.uniform(0.5, 4.5, size=num_samples)
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

    with st.sidebar:
        num_samples = st.number_input(
            "Enter Number of Samples", min_value=3, max_value=300, value=40
        )

    X, Y_obs, Y_true = generate_data(num_samples)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_obs, test_size=0.1, shuffle=True, random_state=42
    )

    n_train = len(X_train)

    X_train_sk = X_train.reshape(-1, 1)
    X_test_sk = X_test.reshape(-1, 1)

    X_plot = np.linspace(np.min(X) - 0.2, np.max(X) + 0.2, 2000)
    X_plot_sk = X_plot.reshape(-1, 1)

    ### sidebar/configure models
    with st.sidebar:
        hide_scatter = st.checkbox("Hide Scatter")
        hide_true_func = st.checkbox("Hide True Function")
        hide_poly = st.checkbox("Hide Polynomial Regression")
        hide_spline = st.checkbox("Hide Spline Regression")

        ## Fit spline
        if not hide_spline:
            st.header("Spline Regression")

            auto_select_knot = st.checkbox("Auto configure knots", value=False)

            if not auto_select_knot:
                st.markdown("### Set Spline Knots")

                # Allow users to dynamically choose how many manual knots to place (max 5)
                num_manual_knots = st.number_input(
                    "Number of Manual Knots", min_value=1, max_value=5, value=2
                )

                manual_knots_list = []

                # Dynamically generate sliders based on the number chosen
                for i in range(num_manual_knots):
                    default_percentile = (i + 1) / (num_manual_knots + 1) * 100
                    default_val = float(np.percentile(X, default_percentile))

                    knot_val = st.slider(
                        rf"Set Knot {i+1} ($\xi_{i+1}$)",
                        min_value=float(np.min(X)),
                        max_value=float(np.max(X)),
                        value=default_val,
                        key=f"knot_slider_{i}",
                    )
                    manual_knots_list.append(knot_val)

                # Combine boundaries with manual knots
                custom_knots = np.sort(
                    [np.min(X)] + manual_knots_list + [np.max(X)]
                ).reshape(-1, 1)

                spline_reg = make_pipeline(
                    SplineTransformer(knots=custom_knots, degree=3, include_bias=True),
                    LinearRegression(fit_intercept=False),
                )
                knot_locs_to_plot = manual_knots_list

            else:
                max_knots = max(2, n_train - 2)
                num_knots = st.number_input(
                    f"Enter Number of Knots (Max {max_knots} for perfect fit)",
                    min_value=2,
                    max_value=max_knots,
                    value=min(5, max_knots),
                )

                spline_reg = make_pipeline(
                    SplineTransformer(
                        n_knots=num_knots, degree=3, knots="quantile", include_bias=True
                    ),
                    LinearRegression(fit_intercept=False),
                )

            spline_reg.fit(X_train_sk, Y_train)
            y_pred_spline_plot = spline_reg.predict(X_plot_sk)

            test_mse_spline = np.round(
                float(mean_squared_error(Y_test, spline_reg.predict(X_test_sk))), 3
            )
            train_mse_spline = np.round(
                float(mean_squared_error(Y_train, spline_reg.predict(X_train_sk))), 3
            )

            if auto_select_knot:
                all_knots = spline_reg.named_steps["splinetransformer"].bsplines_[0].t
                knot_locs_to_plot = np.unique(all_knots)
                knot_locs_to_plot = knot_locs_to_plot[
                    (knot_locs_to_plot >= np.min(X)) & (knot_locs_to_plot <= np.max(X))
                ]

        ## Fit Polynomial Regression
        if not hide_poly:
            st.header("Set Polynomial Degree")

            # Capping the max degree at 40 to prevent Float64 overflow (still shows interpolation to demonstrate overfitting/memorization)
            max_deg = min(40, n_train - 1)
            p = st.number_input(
                f"Set polynomial degree (Capped at 40 for float stability)",
                min_value=1,
                max_value=max_deg,
                step=1,
                value=min(3, max_deg),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                poly_reg = Chebyshev.fit(X_train, Y_train, deg=p)

                raw_pred = poly_reg(X_plot)
                poly_pred_plot = np.nan_to_num(
                    raw_pred, nan=1e10, posinf=1e10, neginf=-1e10
                )

                poly_pred_plot = np.clip(poly_pred_plot, -1e10, 1e10)

                test_mse_poly = np.round(
                    float(mean_squared_error(Y_test, poly_reg(X_test))), 3
                )
                train_mse_poly = np.round(
                    float(mean_squared_error(Y_train, poly_reg(X_train))), 3
                )

    ## Plot

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
                    marker=dict(
                        opacity=1.0,
                        size=10,
                        color="#00BFFF",
                        line=dict(width=1, color="black"),
                    ),
                    name="Train Set",
                )
            )
        if not hide_test:
            fig.add_trace(
                go.Scatter(
                    x=X_test,
                    y=Y_test,
                    mode="markers",
                    marker=dict(
                        opacity=1.0,
                        size=10,
                        color="#F4D03F",
                        symbol="cross",
                        line=dict(width=1, color="black"),
                    ),
                    name="Test Set",
                )
            )

    #### Add true function
    if not hide_true_func:
        X_smooth = np.linspace(0.5, 4.5, 500)
        Y_true_smooth = np.sin(2 * X_smooth) + np.log(10 * X_smooth)

        fig.add_trace(
            go.Scatter(
                x=X_smooth,
                y=Y_true_smooth,
                mode="lines",
                name="True Function",
                line_dash="dash",
                line_width=3,
                line=dict(color="#27ae60"),
            )
        )

    #### Add spline prediction
    if not hide_spline:
        fig.add_trace(
            go.Scatter(
                x=X_plot,
                y=y_pred_spline_plot,
                mode="lines",
                name=f"Spline (Train MSE={train_mse_spline:.3f}, Test MSE={test_mse_spline:.3f})",
                line=dict(color="#8e44ad", width=2),
            )
        )

        with st.sidebar:
            hide_knot = st.checkbox("Hide Knot Location")
        if not hide_knot:
            for i, k_loc in enumerate(knot_locs_to_plot):
                fig.add_vline(
                    x=k_loc,
                    line_color="#A04747",
                    line_dash="dash",
                    opacity=0.4,
                    name=f"knot {i+1}",
                )

    ### Add Polynomial Regression
    if not hide_poly:
        fig.add_trace(
            go.Scatter(
                x=X_plot,
                y=poly_pred_plot,
                mode="lines",
                name=f"Polynomial (Train MSE={train_mse_poly:.3f}, Test MSE={test_mse_poly:.3f})",
                line=dict(color="#d35400", width=2),
            )
        )

    #### Display plot
    y_min_bound = np.min(Y_obs) - 3
    y_max_bound = np.max(Y_obs) + 3

    fig.update_layout(
        width=1200,
        height=600,
        xaxis_title="X",
        yaxis_title="Y",
        yaxis_range=[y_min_bound, y_max_bound],
        hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
