###############################################################################
# Sberbank Russian Housing Market Challenge                                   #
#                                                                             #
# This is the visualizations' generator                                       #
# Developed using Python 3.8.                                                 #
#                                                                             #
# Author: Sarah Zouinina                                                      #
# e-mail: sarahzouinina1@gmail.com                                            #
# Date: 2021-06-09                                                            #
# Version: 1.0.0                                                              #
###############################################################################
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

import matplotlib.pyplot as plt
import seaborn as sns
from src.sberbank_analysis.data_preprocessing import preprocessing_steps
from src.sberbank_analysis.data_training.file_paths import *


def plot(df):
    pr = preprocessing_steps.Preprocessor()
    df = pr.transform(df)
    material_features = ['raion_build_count_with_material_info', 'build_count_block', 'build_count_wood',
                         'build_count_frame',
                         'build_count_brick', 'build_count_monolith', 'build_count_panel', 'build_count_foam',
                         'build_count_slag', 'build_count_mix']
    plt.figure(figsize=(16, 8))
    sns.boxplot(x=df['material'], y=np.log1p(df['price_doc']), data=df[material_features])
    plt.title('The price of the property by the construction material')
    plt.savefig(PLOTS_DIR_str+'/boxp_price_by_material', dpi = 300)

    table = df.groupby(df['sub_area'], as_index=False)['sold_month'].median()
    plt.figure(figsize=(30, 50))
    sns.barplot(x=table['sold_month'], y=table['sub_area'], data=table)
    plt.title('The month each are is mostly sold')
    plt.savefig(PLOTS_DIR_str + '/barp_month_subarea', dpi=300)

    train_monthgrp = df.groupby('sold_year')['price_doc'].aggregate(np.median).reset_index()
    plt.figure(figsize=(24, 16))
    sns.lineplot(x="sold_year", y="price_doc", data=train_monthgrp)
    plt.ylabel('Median Price', fontsize=18)
    plt.xlabel('Year', fontsize=18)
    plt.xticks(rotation='vertical')
    plt.title('The median price by the year the property was sold')
    plt.savefig(PLOTS_DIR_str + '/linep_price_by_year', dpi=300)


    plt.figure(figsize=(16, 8))
    sns.boxplot(x=df['sold_month'], y=np.log1p(df['price_doc']), data=df[material_features])
    plt.title('The price by the month the property was sold')
    plt.savefig(PLOTS_DIR_str + '/boxp_price_by_month', dpi=300)


    plt.figure(figsize=(16, 8))
    sns.boxplot(x=df['sold_day'], y=np.log1p(df['price_doc']), data=df[material_features])
    plt.title('The price by the day the property was sold')
    plt.savefig(PLOTS_DIR_str + '/boxp_price_by_day', dpi=300)


    plt.figure(figsize=(16, 8))
    sns.boxplot(x=df['num_room'], y=np.log1p(df['price_doc']),
                data=df.groupby(df['num_room'], as_index=False)['price_doc'].mean())
    plt.title('Mean price by number of rooms')
    plt.savefig(PLOTS_DIR_str + '/boxp_price_by_room_num', dpi=300)


    plt.figure(figsize=(16, 8))
    sns.boxplot(x=df['material'], y=np.log1p(df['price_doc']),
                data=df.groupby(df['material'], as_index=False)['price_doc'].mean())
    plt.title('Mean price by material')
    plt.savefig(PLOTS_DIR_str + '/boxp_mean_price_by_material', dpi=300)








