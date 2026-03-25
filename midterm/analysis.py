from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / ".runtime"
RUNTIME_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_DIR / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_DIR / "cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy.stats import bartlett
from semopy import Model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

OUTPUT_DIR = BASE_DIR


def configure_chinese_font():
    candidates = [
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Serif CJK JP",
        "Noto Serif CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "WenQuanYi Zen Hei",
        "Droid Sans Fallback",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((font for font in candidates if font in available), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


sns.set_theme(style="whitegrid")
SELECTED_FONT = configure_chinese_font()
plt.rcParams["font.family"] = "sans-serif"


def save_current_figure(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def load_data():
    df = pd.read_excel(BASE_DIR / "问卷.xlsx", sheet_name="AI数字人满意度调查")
    df_clean = df.iloc[1:].reset_index(drop=True)
    df_clean.columns = [
        "填写人", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10",
        "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18", "Q19", "Q20",
        "Q21", "Q22", "Q23", "Q24", "Q25", "Q26", "空1", "空2", "性别", "年龄段",
        "重点扶持村", "参与形式", "接触场景",
    ]
    df_clean = df_clean.drop(["空1", "空2"], axis=1)

    question_cols = [f"Q{i}" for i in range(1, 27)]
    for col in question_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    for col in ["性别", "年龄段", "重点扶持村"]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    df_clean["技术赋权"] = df_clean[[f"Q{i}" for i in range(1, 9)]].mean(axis=1)
    df_clean["经济赋权"] = df_clean[[f"Q{i}" for i in range(9, 17)]].mean(axis=1)
    df_clean["话语赋权"] = df_clean[[f"Q{i}" for i in range(17, 25)]].mean(axis=1)
    df_clean["满意度"] = df_clean[["Q25", "Q26"]].mean(axis=1)
    return df_clean, question_cols


def cronbach_alpha(df, items):
    items = df[items].dropna()
    n = items.shape[1]
    total_var = items.var(axis=0, ddof=1).sum()
    total_score = items.sum(axis=1)
    score_var = total_score.var(ddof=1)
    return (n / (n - 1)) * (1 - total_var / score_var)


def plot_mean_scores(mean_scores):
    plt.figure(figsize=(12, 8))
    color_positions = np.linspace(0, 1, len(mean_scores))
    cmap = LinearSegmentedColormap.from_list(
        "score_gradient",
        ["#103b73", "#2382c4", "#6ac7d3", "#f7c873"],
    )
    colors = cmap(color_positions)

    bars = plt.barh(mean_scores.index, mean_scores.values, color=colors, edgecolor="#1f2a44", linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel("平均分")
    plt.ylabel("题项")
    plt.title("各题项平均分排序")
    plt.xlim(0, max(5, mean_scores.max() + 0.15))

    for bar, value in zip(bars, mean_scores.values):
        plt.text(
            value + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=9,
            color="#1f2a44",
        )

    save_current_figure("图1_各题项平均分排序.png")


def plot_correlation_heatmap(df_clean, question_cols):
    plt.figure(figsize=(12, 9))
    sns.heatmap(df_clean[question_cols].corr(), annot=False, cmap="YlGnBu", square=True, cbar_kws={"shrink": 0.8})
    plt.title("问卷题目相关性热图", fontsize=14)
    save_current_figure("图2_问卷题目相关性热图.png")


def plot_boxplot(df_clean, question_cols):
    plt.figure(figsize=(14, 7))
    palette = sns.color_palette("Spectral", n_colors=len(question_cols))
    sns.boxplot(data=df_clean[question_cols], orient="h", palette=palette, linewidth=1)
    plt.title("各题得分箱线图")
    plt.xlabel("得分分布")
    save_current_figure("图3_各题得分箱线图.png")


def plot_radar_chart(df_clean):
    labels = ["技术赋权", "经济赋权", "话语赋权"]
    values = [
        df_clean["技术赋权"].mean(),
        df_clean["经济赋权"].mean(),
        df_clean["话语赋权"].mean(),
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, "o-", linewidth=2.5, color="#0b6e99")
    ax.fill(angles, values, alpha=0.25, color="#57c4c6")
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_ylim(0, 5)
    ax.set_title("三类赋权平均得分雷达图", pad=20)
    save_current_figure("图12_三类赋权平均得分雷达图.png")


def plot_factor_loading_heatmap(loadings):
    plt.figure(figsize=(10, 7))
    sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0)
    plt.title("因子载荷矩阵热图（前3因子）")
    plt.xlabel("因子")
    plt.ylabel("题目编号")
    save_current_figure("图5_因子载荷矩阵热图.png")


def plot_feature_importance(xgb_model, columns):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=xgb_model.feature_importances_, y=columns, hue=columns, palette="crest", dodge=False, legend=False)
    plt.title("XGBoost 特征重要性")
    plt.xlabel("重要性")
    plt.ylabel("变量")
    save_current_figure("图6_XGBoost特征重要性.png")


def plot_regression_relationships(df_clean):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#0b6e99", "#e58e26", "#7c4dff"]
    for ax, var, color in zip(axes, ["技术赋权", "经济赋权", "话语赋权"], colors):
        sns.regplot(
            x=var,
            y="满意度",
            data=df_clean,
            ax=ax,
            color=color,
            scatter_kws={"alpha": 0.7, "s": 45},
            line_kws={"color": "#c0392b", "linewidth": 2},
        )
        ax.set_title(f"{var} 与 满意度关系")
    save_current_figure("图7_赋权因素与满意度回归关系.png")


def plot_residuals(model):
    residuals = model.resid
    fitted = model.fittedvalues
    plt.figure(figsize=(8, 6))
    sns.residplot(x=fitted, y=residuals, lowess=True, color="#2563eb")
    plt.axhline(0, color="#dc2626", linestyle="--", linewidth=1.5)
    plt.title("OLS 残差图")
    plt.xlabel("拟合值")
    plt.ylabel("残差")
    save_current_figure("图8_OLS残差图.png")


def plot_coefficients(model):
    coef_df = pd.DataFrame({
        "变量": model.params.index[1:],
        "系数": model.params.values[1:],
    }).sort_values("系数", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x="系数", y="变量", data=coef_df, hue="变量", palette="mako", dodge=False, legend=False)
    plt.title("各赋权因素对满意度的影响系数")
    save_current_figure("图9_赋权因素影响系数.png")


def plot_prediction_scatter(y_test, y_pred):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, color="#8e44ad", s=55, edgecolor="white", linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=1.5)
    plt.xlabel("实际满意度")
    plt.ylabel("预测满意度")
    plt.title("XGBoost 实际值 vs 预测值")
    save_current_figure("图10_XGBoost实际值_vs_预测值.png")


def plot_error_distribution(y_test, y_pred):
    errors = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(errors, kde=True, color="#f39c12")
    plt.title("预测误差分布（XGBoost）")
    plt.xlabel("误差值")
    save_current_figure("图11_XGBoost预测误差分布.png")


def plot_sem_diagram(sem_results):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    positions = {
        "技术赋权": (0.18, 0.78),
        "经济赋权": (0.18, 0.50),
        "话语赋权": (0.18, 0.22),
        "满意度": (0.78, 0.50),
    }
    node_styles = {
        "技术赋权": {"fc": "#d9eefc", "ec": "#0b6e99"},
        "经济赋权": {"fc": "#fde8c8", "ec": "#d97706"},
        "话语赋权": {"fc": "#e6dcff", "ec": "#7c3aed"},
        "满意度": {"fc": "#d8f3dc", "ec": "#2f855a"},
    }

    for node, (x, y) in positions.items():
        style = node_styles[node]
        ax.text(
            x,
            y,
            node,
            ha="center",
            va="center",
            fontsize=17,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": style["fc"],
                "edgecolor": style["ec"],
                "linewidth": 2,
            },
        )

    sem_lookup = sem_results[sem_results["op"] == "~"].set_index("rval")
    arrow_specs = [
        ("技术赋权", "满意度", 0.08, (0.00, 0.17), (0.00, 0.07)),
        ("经济赋权", "满意度", 0.00, (0.00, 0.12), (0.00, 0.03)),
        ("话语赋权", "满意度", -0.08, (0.00, -0.17), (0.00, -0.07)),
    ]

    for source, target, rad, label_offset, est_offset in arrow_specs:
        sx, sy = positions[source]
        tx, ty = positions[target]
        est = sem_lookup.loc[source, "Estimate"]
        std_est = sem_lookup.loc[source, "Est. Std"]
        p_value = sem_lookup.loc[source, "p-value"]
        color = "#2f855a" if float(p_value) < 0.05 else "#7f8c8d"

        arrow = FancyArrowPatch(
            (sx + 0.08, sy),
            (tx - 0.1, ty),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.4,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(arrow)

        base_x = (sx + tx) / 2 - 0.02
        base_y = (sy + ty) / 2
        label_x = base_x + label_offset[0]
        label_y = base_y + label_offset[1]
        ax.text(
            label_x,
            label_y,
            f"β={std_est:.3f}\np={float(p_value):.3g}",
            fontsize=13,
            ha="center",
            va="center",
            color=color,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": color, "alpha": 0.95},
        )
        ax.text(
            base_x + est_offset[0],
            base_y + est_offset[1],
            f"非标准化={est:.3f}",
            fontsize=11,
            ha="center",
            va="center",
            color="#374151",
        )

    residual = sem_results[(sem_results["lval"] == "满意度") & (sem_results["op"] == "~~")]["Estimate"].iloc[0]
    ax.text(0.78, 0.15, f"残差方差={residual:.3f}", ha="center", fontsize=13, color="#374151")
    ax.set_title("图4 结构方程模型路径图", fontsize=18, pad=18)
    save_current_figure("图4_SEM路径图.png")


def main():
    print(f"图表输出目录: {OUTPUT_DIR}")
    print(f"当前中文字体: {SELECTED_FONT}")

    df_clean, question_cols = load_data()

    print("\n=== 描述性统计分析 ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    desc_stats = df_clean[question_cols].describe()
    print(desc_stats)

    mean_scores = df_clean[question_cols].mean().sort_values(ascending=False)
    print("\n各题平均分排序：")
    for i, (q, score) in enumerate(mean_scores.items(), 1):
        print(f"{i:2d}. {q}: {score:.3f}")

    plot_mean_scores(mean_scores)
    plot_correlation_heatmap(df_clean, question_cols)
    plot_boxplot(df_clean, question_cols)
    plot_radar_chart(df_clean)

    print("\n=== 信效度分析 ===")
    alpha = cronbach_alpha(df_clean, question_cols)
    print(f"克朗巴哈α系数: {alpha:.3f}")

    kmo_all, kmo_model = calculate_kmo(df_clean[question_cols].dropna())
    print(f"KMO 值: {kmo_model:.3f}（>0.7 表示适合做因子分析）")

    chi_square_value, p_value = bartlett(*[df_clean[col].dropna() for col in question_cols])
    print(f"Bartlett球形检验: χ²={chi_square_value:.2f}, p值={p_value:.4f}")

    fa = FactorAnalyzer(n_factors=3, rotation="varimax")
    fa.fit(df_clean[question_cols].dropna())
    loadings = fa.loadings_
    plot_factor_loading_heatmap(loadings)

    print("\n=== 多元回归分析 ===")
    X = df_clean[["技术赋权", "经济赋权", "话语赋权"]]
    y = df_clean["满意度"]
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const, missing="drop").fit()
    print(model.summary())

    print("\n回归结果解读：")
    print("若显著性水平 p < 0.05，则表明该赋权因素对满意度具有显著影响。")
    print("R² 值越高，模型解释力越强。")

    print("\n=== XGBoost 回归分析 ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"R² = {r2:.3f}, 均方误差 = {mse:.4f}")

    plot_feature_importance(xgb_model, X_train.columns)
    plot_regression_relationships(df_clean)
    plot_residuals(model)
    plot_coefficients(model)
    plot_prediction_scatter(y_test, y_pred)
    plot_error_distribution(y_test, y_pred)

    print("\n=== 结构方程模型（SEM）分析 ===")
    model_desc = """
    满意度 ~ 技术赋权 + 经济赋权 + 话语赋权
    """
    sem_df = df_clean[["满意度", "技术赋权", "经济赋权", "话语赋权"]].dropna()
    sem_model = Model(model_desc)
    sem_model.fit(sem_df)
    sem_results = sem_model.inspect(std_est=True)
    print(sem_results)
    plot_sem_diagram(sem_results)

    cleaned_path = OUTPUT_DIR / "清洗后的问卷数据_分析版.xlsx"
    df_clean.to_excel(cleaned_path, index=False)
    print(f"\n清洗并分析后的数据已保存为: {cleaned_path}")


if __name__ == "__main__":
    main()
