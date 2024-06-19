import pandas as pd
import os


def group_and_average_to_excel(output_folder, num_iterations):
    try:
        # Read Excel file

        # Check if Pathology Number, Distance from Border, Importance Order, Number of Sections, and Label exist
        required_columns = ["pathology num", distance_name, importance_name, "patch num"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"Columns {', '.join(required_columns)} are required in the Excel sheet."
            )

        # Sort by Importance in ascending order
        df.sort_values(
            by=["pathology num", importance_name], inplace=True, na_position="last"
        )
        result_df = pd.DataFrame()

        for i in range(num_iterations):
            # Modify iloc parameters to determine the step size
            end_index = (i + 1) * 5
            top_average = (
                df.groupby("pathology num")
                .apply(lambda group: group.head(end_index)[distance_name].mean())
                .reset_index(name=f"Top {end_index} {distance_name} mean")
            )

            # Merge into the result_df
            result_df = pd.concat([result_df, top_average.set_index("pathology num")], axis=1)

        # Calculate the overall average and count, including rows where Importance Order is null
        overall_stats = (
            df.groupby("pathology num")[distance_name].agg(["mean", "count"]).reset_index()
        )
        overall_stats.columns = ["pathology num", "all patch mean", "all num"]

        # Merge into the result_df
        result_df = pd.merge(result_df, overall_stats, on="pathology num")

        # Calculate the average of the number of sections
        additional_cols_average = df.groupby("pathology num")["patch num"].mean().reset_index()


        # Merge the average number of sections and the first column value of the Label
        result_df = pd.merge(result_df, additional_cols_average, on="pathology num")

        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Construct the output file path
        output_file = (
            f"{output_folder}/{file_name}_{importance_name}_{distance_name}_mean result.xlsx"
        )

        # Output to a separate Excel file
        result_df.to_excel(output_file, index=False)

        print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Set file path
file_path = r"C:\Users\32618\Desktop\distance(table S5)\distance data\iPM Stroma.xlsx"#data from distance data
sheet_name = "Sheet1"
output_folder = r"C:\Users\32618\Desktop\distance(table S5)\Top patch average distance"
num_iterations = 10
importance_group = [
    "model ranking"
]
distance_group = ["patch distance"]
df = pd.read_excel(file_path, sheet_name=sheet_name)
print("read over")
for importance_name in importance_group:
    for distance_name in distance_group:
        group_and_average_to_excel(output_folder, num_iterations)
