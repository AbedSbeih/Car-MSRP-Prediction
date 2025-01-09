{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 3575,
     "status": "ok",
     "timestamp": 1668389926795,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "flUNma5syFmo",
    "outputId": "af0490e3-b06a-4b83-f94b-fa58b240b94e"
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "# Load data\n",
    "from google.colab import drive\n",
    "drive.mount('/content/gdrive')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "JvFnb_i0vDKa"
   },
   "outputs": [],
   "source": [
    "data = pd.read_csv('/content/gdrive/MyDrive/cars (1).csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 124,
     "status": "ok",
     "timestamp": 1668389927561,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "C7M_mTw2vOXu",
    "outputId": "eb12f6da-2927-49b3-a944-a04b6a49f144"
   },
   "outputs": [],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 832
    },
    "executionInfo": {
     "elapsed": 120,
     "status": "ok",
     "timestamp": 1668389927562,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "AeicnltSvSc0",
    "outputId": "4a6e87e1-4b08-429b-dc77-947cfe559615"
   },
   "outputs": [],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "7K1ZbngCoTGS"
   },
   "outputs": [],
   "source": [
    "X=data.copy()\n",
    "y=X.pop('MSRP')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 693
    },
    "executionInfo": {
     "elapsed": 114,
     "status": "ok",
     "timestamp": 1668389927565,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "LzfW_v8HoeF5",
    "outputId": "71172fdc-90d9-4c03-df86-da7b21f53ce8"
   },
   "outputs": [],
   "source": [
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 113,
     "status": "ok",
     "timestamp": 1668389927566,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "6wBJPOLfois7",
    "outputId": "9f7abe43-38c2-44cb-beff-6a424828c1b6"
   },
   "outputs": [],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 105,
     "status": "ok",
     "timestamp": 1668389927568,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "9uwSyaoCojXI",
    "outputId": "28e437a4-8fb5-4fb5-9cf2-1e8f1c26d085"
   },
   "outputs": [],
   "source": [
    "X['Vehicle Size'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "NrCW2S6G8aLq"
   },
   "source": [
    "First of all I will handle the missing values using imputation with different strategies :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 101,
     "status": "ok",
     "timestamp": 1668389927569,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "sPb91lSbophd",
    "outputId": "a3e470ff-fdd5-449f-a5e1-e17b81c15f86"
   },
   "outputs": [],
   "source": [
    "\n",
    "cols_with_missing=[col for col in X if X[col].isnull().any()]\n",
    "\n",
    "cols_with_missing\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 97,
     "status": "ok",
     "timestamp": 1668389927570,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "f1Z8AxiGqA0w",
    "outputId": "1d885722-4733-44b4-d2ca-c940519461d9"
   },
   "outputs": [],
   "source": [
    "X[cols_with_missing].dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "Gcudm3uJxlfH"
   },
   "outputs": [],
   "source": [
    "numeric_x=X.select_dtypes(exclude=['object'])\n",
    "\n",
    "from sklearn.impute import SimpleImputer\n",
    "\n",
    "my_imputer1=SimpleImputer(strategy='mean')\n",
    "numeric_x_imputed=pd.DataFrame(my_imputer1.fit_transform(numeric_x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "nGeX-Nso1Ex9"
   },
   "outputs": [],
   "source": [
    "categorical_x=X.select_dtypes(exclude=['int64','float64'])\n",
    "\n",
    "my_imputer2=SimpleImputer(strategy='most_frequent')\n",
    "categorical_x_imputed=pd.DataFrame(my_imputer2.fit_transform(categorical_x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "JrlTcyle8G8l"
   },
   "outputs": [],
   "source": [
    "numerical_cols=[col for col in X if X[col].dtypes=='float64' or X[col].dtypes=='int64']\n",
    "categorical_cols=[col for col in X if X[col].dtypes=='object']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "1ZZoKPxp2kYR"
   },
   "outputs": [],
   "source": [
    "numeric_x_imputed.columns=numerical_cols\n",
    "categorical_x_imputed.columns=categorical_cols"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 424
    },
    "executionInfo": {
     "elapsed": 121,
     "status": "ok",
     "timestamp": 1668389930838,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "OjtAbjLK3Ay6",
    "outputId": "b88c9c15-30f4-4198-a259-8da81d806b16"
   },
   "outputs": [],
   "source": [
    "numeric_x_imputed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 424
    },
    "executionInfo": {
     "elapsed": 117,
     "status": "ok",
     "timestamp": 1668389930839,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "l7ItXGj74cgl",
    "outputId": "e63a4aed-fa6b-47df-f320-2ce4c77f0d0e"
   },
   "outputs": [],
   "source": [
    "categorical_x_imputed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "LVFODzKM5TUK"
   },
   "outputs": [],
   "source": [
    "X_new=pd.concat([numeric_x_imputed,categorical_x_imputed],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 693
    },
    "executionInfo": {
     "elapsed": 114,
     "status": "ok",
     "timestamp": 1668389930843,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "ViIQKEFz5ia6",
    "outputId": "382f43cc-e5d3-4220-b149-bdab8a7b35b9"
   },
   "outputs": [],
   "source": [
    "X_new"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 110,
     "status": "ok",
     "timestamp": 1668389930844,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "u0Odtx-O7EUi",
    "outputId": "9326d14e-606d-4ddb-f180-a66bfd4d5c78"
   },
   "outputs": [],
   "source": [
    "X_new.isnull().sum().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "_pUMiNnA-FuV"
   },
   "source": [
    "After filling the missing values , I will use some feature engineering aproaches ,starting with finding the MI :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "Qtsv5n9q7fvy"
   },
   "outputs": [],
   "source": [
    "from sklearn.feature_selection import mutual_info_regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "m5hoZtOo99d9"
   },
   "outputs": [],
   "source": [
    "\n",
    "for colname in X_new.select_dtypes(\"object\"):\n",
    "    X_new[colname], _ = X_new[colname].factorize()\n",
    "\n",
    "discrete_features = X_new.dtypes == int\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "MPg8tu_l-AXw"
   },
   "outputs": [],
   "source": [
    "mi_scores = mutual_info_regression(X_new, y, discrete_features=discrete_features)\n",
    "mi_scores = pd.Series(mi_scores, name=\"MI Scores\", index=X_new.columns)\n",
    "mi_scores = mi_scores.sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 49,
     "status": "ok",
     "timestamp": 1668389932938,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "ih2w4FJx_n4C",
    "outputId": "0c394c77-f3ef-4515-9cb0-884d340ab750"
   },
   "outputs": [],
   "source": [
    "mi_scores\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "eeAQ70u8_qoR"
   },
   "outputs": [],
   "source": [
    "# I can create new feature here , I searched fow a way to combine city MPG and highway MPG\n",
    "# so I will add a new feature\n",
    "# you can ask a domain expert to find the formula, in my case I used Google\n",
    "\n",
    "\n",
    "X_new['Combined_MPG']=  X_new['city mpg'] * .55 +  X_new['highway MPG'] * .45"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 24,
     "status": "ok",
     "timestamp": 1668389932941,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "YdDKbKTJ_vL2",
    "outputId": "7a9821da-64f8-40e7-fa7b-386b7e14320f"
   },
   "outputs": [],
   "source": [
    "X_new['Combined_MPG']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 3269,
     "status": "ok",
     "timestamp": 1668389936193,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "-hciPoSlIV9O",
    "outputId": "45aca0fe-9414-4ad9-f5d4-660c047ff8f9"
   },
   "outputs": [],
   "source": [
    "# I will do new MI serie, if my new feature gets high MI , I will keep it\n",
    "\n",
    "for colname in X_new.select_dtypes(\"object\"):\n",
    "    X_new[colname], _ = X_new[colname].factorize()\n",
    "\n",
    "discrete_features = X_new.dtypes == int\n",
    "\n",
    "\n",
    "\n",
    "mi_scores = mutual_info_regression(X_new, y, discrete_features=discrete_features)\n",
    "mi_scores = pd.Series(mi_scores, name=\"MI Scores\", index=X_new.columns)\n",
    "mi_scores = mi_scores.sort_values(ascending=False)\n",
    "\n",
    "\n",
    "mi_scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "D5Eb7852G9oJ"
   },
   "outputs": [],
   "source": [
    "# another thing I can do is to remove 'number of doors' feature since it has the least MI\n",
    "# which makes sense because you can conclude the number of doors by other features 'vehicle size' and 'vehicle style'\n",
    "\n",
    "\n",
    "X_new.drop('Number of Doors',axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 485
    },
    "executionInfo": {
     "elapsed": 56,
     "status": "ok",
     "timestamp": 1668389936195,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "Iwbp-EwqHAwN",
    "outputId": "2fa502d4-b7cb-4015-c9eb-be7da88a7949"
   },
   "outputs": [],
   "source": [
    "X_new"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "gkbufKaLTDVL"
   },
   "source": [
    "Now I will build the model and then use MAE for validation :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "W7AeOyagQi9e"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "DkH9eE9iQjFN"
   },
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "from sklearn.model_selection import train_test_split\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "ZbHyrUNdZGdy"
   },
   "outputs": [],
   "source": [
    "def get_MAE(e):\n",
    "  model=RandomForestRegressor(n_estimators=e,random_state=0)\n",
    "  X_train,X_val,y_train,y_val=train_test_split(X_new,y,train_size=.8,test_size=.2,random_state=0)\n",
    "  model.fit(X_train,y_train)\n",
    "  yp=model.predict(X_val)\n",
    "  error=mean_absolute_error(yp,y_val)\n",
    "  return error\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "9NmQLvjmi4Ro"
   },
   "source": [
    "Now I will try the model parameters to find the least error   :\n",
    "\n",
    "(I will try different number of estimators and then validate the model using the cross validation function)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 5758,
     "status": "ok",
     "timestamp": 1668389941906,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "DPVTbH_vd-ge",
    "outputId": "b072581f-e253-4e92-acde-8fd1e4cd0483"
   },
   "outputs": [],
   "source": [
    "get_MAE(50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 13582,
     "status": "ok",
     "timestamp": 1668389955477,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "swCCUCEed-sm",
    "outputId": "10f8d23d-05b1-41f7-a4b3-af678e0c9781"
   },
   "outputs": [],
   "source": [
    "for i in [25,50,75,100]:\n",
    "  print(get_MAE(i))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 27209,
     "status": "ok",
     "timestamp": 1668389982676,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "cYMdjps-d-2p",
    "outputId": "a03208ee-8762-4a8e-99b6-c964ebdd89ca"
   },
   "outputs": [],
   "source": [
    "for i in [100,200,300]:\n",
    "  print(get_MAE(i))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 49038,
     "status": "ok",
     "timestamp": 1668390031706,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "7NspX9kyfL0n",
    "outputId": "2be0e300-8f5b-4832-b951-ada74cf26a7a"
   },
   "outputs": [],
   "source": [
    "for i in [300,350,400]:\n",
    "  print(get_MAE(i))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 20384,
     "status": "ok",
     "timestamp": 1668390051895,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "8XcE5Q7Gg5IH",
    "outputId": "ef98280e-97ec-4c82-9628-48d1f5f52516"
   },
   "outputs": [],
   "source": [
    "#The last MAE before creating the final model\n",
    "get_MAE(400)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "D8L90cdoim5d"
   },
   "outputs": [],
   "source": [
    "# The best number of estimators is 400"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5aUEMhFell1i"
   },
   "source": [
    "Now,I can build my final model and train it in all of the data !"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 21641,
     "status": "ok",
     "timestamp": 1668390073528,
     "user": {
      "displayName": "Abed Sbeih",
      "userId": "16591061846236117142"
     },
     "user_tz": -180
    },
    "id": "Tv_z-l9Tj9Ud",
    "outputId": "c6ba6253-6d6b-45ec-c337-501784aedfe0"
   },
   "outputs": [],
   "source": [
    "\n",
    "myFinalModel=RandomForestRegressor(n_estimators=400,random_state=0)\n",
    "myFinalModel.fit(X_new,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "N0jEos8L2969"
   },
   "outputs": [],
   "source": [
    "import pickle\n",
    "\n",
    "filename = 'finalized_model.sav'\n",
    "pickle.dump(myFinalModel, open(filename, 'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "WDTP6EVd2-Db"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "wRucQGZq2-Kb"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "hnWrpTD32-Rm"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
