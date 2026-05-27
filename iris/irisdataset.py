# 保存为 save_iris_to_excel.py 然后运行：python save_iris_to_excel.py
import pandas as pd
from sklearn.datasets import load_iris


def main(output_path="iris.xlsx"):
    # 加载iris数据
    iris = load_iris(as_frame=True)
    X = iris.data  # 特征DataFrame，列名是feature名称
    y = iris.target  # 标签Series（0/1/2）
    target_names = iris.target_names

    # 将数字标签映射为物种名称，新增一列 species
    species = y.map({i: name for i, name in enumerate(target_names)})

    # 合并为一个完整DataFrame
    df = X.copy()
    df["target"] = y
    df["species"] = species

    # 可选：顺序调整
    cols = list(X.columns) + ["target", "species"]
    df = df[cols]

    # 保存为Excel
    df.to_excel(output_path, index=False)
    print(f"已保存到: {output_path}")


if __name__ == "__main__":
    main()
