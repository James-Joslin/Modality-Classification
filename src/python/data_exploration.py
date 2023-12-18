import json
import util
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.stats import zscore


if __name__ == "__main__":
    # Personal info - secured by a secrets .json file kept out of the repo by the .gitignore
    with open('secrets.json', 'r') as file:
        secrets = json.load(file)
    file.close()
    connection_params = {
        "username": secrets['sqlConfig']['username'],
        "password": secrets['sqlConfig']['password'],
        "port": secrets['sqlConfig']['port'],
        "node_ip": secrets['sqlConfig']['ip'],
        "db_name": secrets['sqlConfig']['db_name']
    }
    
    df = util.get_data(**connection_params, query_path=secrets["sqlQueryFilePaths"]["all_modality_data"])
    print(df['unique_referral_index'].unique())
    df['opened_year'] = df['opened_date'].dt.year
    df['opened_month'] = df['opened_date'].dt.month
    df['opened_day'] = df['opened_date'].dt.day

    df['assessment_year'] = df['first_assessment_date_offered'].dt.year
    df['assessment_month'] = df['first_assessment_date_offered'].dt.month
    df['assessment_day'] = df['first_assessment_date_offered'].dt.day

    # Step 2: Calculate the time difference in days
    df['wait_time_days'] = (df['first_assessment_date_offered'] - df['opened_date']).dt.days
    
    # Step 3: Filter out records with bad dates
    df = df[df['wait_time_days'] >= 0]
    zscore_threshold = 2.5
    df['wait_time_zscore'] = zscore(df['wait_time_days'])
    # Filter out records with a Z-score beyond the threshold
    df = df[df['wait_time_zscore'].abs() <= zscore_threshold]
    df['treatment_time_days'] = (df['discharge_date'] - df['first_assessment_date_offered']).dt.days
    # Filter out records with negative time difference
    df = df[df['treatment_time_days'] >= 0]
    df['treatment_time_zscore'] = zscore(df['treatment_time_days'])
    # Filter out records with a Z-score beyond the threshold
    df = df[df['treatment_time_zscore'].abs() <= zscore_threshold]
    
    # Step 3: CLean modality data        
    # Create an instance of ModalityEncoder, fit it, and transform the DataFrame
    modality_encoder = util.InitialModalityEncoder('modality_type')
    modality_encoder.fit(df)
    df = modality_encoder.transform(df)
    print(df.columns)
    # Get the lookup table
    modality_lookup_table = modality_encoder.get_lookup_table()

    # Assuming df_final is your original DataFrame
    # Create a long-form dataset for opened and assessment dates
    opened_df = df[['opened_month', 'opened_year', 'first_pgsi', 'first_core10', 'last_max_pgsi', 'last_max_core10']].copy()
    opened_df.rename(columns={'opened_month': 'month', 'opened_year': 'year'}, inplace=True)
    opened_df['date_type'] = 'Opened'

    assessment_df = df[['assessment_month', 'assessment_year', 'first_pgsi', 'first_core10', 'last_max_pgsi', 'last_max_core10']].copy()
    assessment_df.rename(columns={'assessment_month': 'month', 'assessment_year': 'year'}, inplace=True)
    assessment_df['date_type'] = 'Assessment'

    # Calculate the differences in scores
    df['pgsi_change'] = df['last_max_pgsi'] - df['first_pgsi']
    df['core10_change'] = df['last_max_core10'] - df['first_core10']
    
    # Concatenate both dataframes
    long_df = pd.concat([opened_df, assessment_df], ignore_index=True)

    sns.set_palette('pastel')
    util.time_metric_boxplot('month', 'first_pgsi', 'first_core10', 'date_type', long_df)
    util.time_metric_boxplot('year', 'first_pgsi', 'first_core10', 'date_type', long_df)
    util.time_metric_boxplot('month', 'last_max_pgsi', 'last_max_core10', 'date_type', long_df)
    util.time_metric_boxplot('year', 'last_max_pgsi', 'last_max_core10', 'date_type', long_df)
    
    util.time_scatterplot('wait_time_days', 'pgsi_change', 'core10_change', df)
    util.time_scatterplot('treatment_time_days', 'pgsi_change', 'core10_change', df)
    
    # Convert the lookup DataFrame to a dictionary
    lookup_dict = modality_lookup_table.set_index('element_id')['first_element'].to_dict()

    # Map modality_type_int to labels in your main DataFrame
    df['modality_label'] = df['first_element_int'].map(lookup_dict)

    plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(1, 2, 1)
    sns.boxplot(y='first_pgsi', x='modality_type_int', hue='modality_label', data=df)
    ax1.set_ylabel('First PGSI')
    ax1.set_xlabel('Modality Type')
    ax1.get_legend().remove()  # Remove the individual plot's legend

    ax2 = plt.subplot(1, 2, 2)
    sns.boxplot(y='first_core10', x='modality_type_int', hue='modality_label', data=df)
    ax2.set_ylabel('First CORE10')
    ax2.set_xlabel('Modality Type')
    ax2.get_legend().remove() 

    # Creating a custom legend
    # Extract the handles and labels from one of the plots
    handles, labels = plt.subplot(1, 2, 1).get_legend_handles_labels()

    # Place a single legend below the subplots
    plt.figlegend(handles, labels, loc='lower center', ncol=3, title='Modality')

    # Adjust subplot parameters for better fit
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()

    plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(1, 2, 1)
    sns.boxplot(y='last_max_pgsi', x='modality_type_int', hue='modality_label', data=df)
    ax1.set_ylabel('Last Max PGSI')
    ax1.set_xlabel('Modality Type')
    ax1.get_legend().remove()  # Remove the individual plot's legend

    ax2 = plt.subplot(1, 2, 2)
    sns.boxplot(y='last_max_core10', x='modality_type_int', hue='modality_label', data=df)
    ax2.set_ylabel('Last Max CORE10')
    ax2.set_xlabel('Modality Type')
    ax2.get_legend().remove() 

    # Creating a custom legend
    # Extract the handles and labels from one of the plots
    handles, labels = plt.subplot(1, 2, 1).get_legend_handles_labels()

    # Place a single legend below the subplots
    plt.figlegend(handles, labels, loc='lower center', ncol=3, title='Modality')

    # Adjust subplot parameters for better fit
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()
    
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x='last_max_pgsi', y='last_max_core10', hue = 'modality_label', data=df, s = 3.5)
    plt.title('Relationship between Last Max PGSI and Core 10 clustered by Modality')
    plt.xlabel('Last Max PGSI')
    plt.ylabel('Last Max Core10')

    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert modality labels to categorical type and get codes
    categories = pd.Categorical(df['modality_label'])
    codes = categories.codes

    # Scatter plot
    scatter = ax.scatter(df['first_pgsi'], df['first_core10'], df['unique_referral_index'],
                        c=codes, s=3.5)  # Using a colormap

    # Create a custom legend
    # Get unique labels and their corresponding colors from the colormap
    unique_labels = categories.categories
    unique_colors = [scatter.cmap(scatter.norm(code)) for code in range(len(unique_labels))]

    # Create legend entries
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                markerfacecolor=color, markersize=10)
                    for label, color in zip(unique_labels, unique_colors)]
    ax.legend(handles=legend_elements, title="Modalities")

    # Labels and title
    ax.set_xlabel('First PGSI')
    ax.set_ylabel('First Core10')
    ax.set_zlabel('Referral Index')
    ax.set_title('First PGCS/Core10/Referral Clustered by Modality')

    # Show plot
    plt.tight_layout()
    plt.show()

    # Step 4: Drop all other columns except the specified ones
    columns_to_keep = ['first_pgsi', 'last_max_pgsi', 'first_core10', 'last_max_core10',
                    'opened_year', 'opened_month', 'opened_day',
                    'assessment_year', 'assessment_month', 'assessment_day', 
                    'wait_time_days', 'treatment_time_days', 'unique_referral_index',
                    'merged_referral_index', 'last_max_core10_bracket', 'last_max_pgsi_bracket', 'modality_type_int']    
    df_final = df[columns_to_keep].reset_index(drop=True)
    print(len(df))
    print(len(df_final))

    corr = df_final.corr()

    # Create a heatmap to visualize the correlation matrix
    plt.figure(figsize=(10, 8))  # You can adjust the size of the figure
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()