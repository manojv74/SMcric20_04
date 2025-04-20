import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns

class IPLWinPredictor:
    def __init__(self):
        self.le_team = LabelEncoder()
        self.le_city = LabelEncoder()
        self.le_venue = LabelEncoder()
        self.le_toss_decision = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.known_teams = None
        self.known_cities = None
        self.known_venues = None
        self.known_toss_decisions = None
        
    def fit_encoders(self, df):
        self.known_teams = sorted(set(df['team1'].unique()) | set(df['team2'].unique()))
        self.known_cities = sorted(df['city'].unique())
        self.known_venues = sorted(df['venue'].unique())
        self.known_toss_decisions = sorted(df['toss_decision'].unique())
        
        self.le_team.fit(self.known_teams)
        self.le_city.fit(self.known_cities)
        self.le_venue.fit(self.known_venues)
        self.le_toss_decision.fit(self.known_toss_decisions)
    
    def encode_with_unknown(self, series, encoder, known_values):
        series = series.map(lambda x: known_values[0] if x not in known_values else x)
        return encoder.transform(series)
    
    def prepare_features(self, df, is_training=True):
        if is_training:
            self.fit_encoders(df)
        
        df_encoded = df.copy()
        
        try:
            # Basic encoding
            df_encoded['team1_encoded'] = self.encode_with_unknown(df['team1'], self.le_team, self.known_teams)
            df_encoded['team2_encoded'] = self.encode_with_unknown(df['team2'], self.le_team, self.known_teams)
            df_encoded['city_encoded'] = self.encode_with_unknown(df['city'], self.le_city, self.known_cities)
            df_encoded['venue_encoded'] = self.encode_with_unknown(df['venue'], self.le_venue, self.known_venues)
            df_encoded['toss_winner_encoded'] = self.encode_with_unknown(df['toss_winner'], self.le_team, self.known_teams)
            df_encoded['toss_decision_encoded'] = self.encode_with_unknown(df['toss_decision'], self.le_toss_decision, self.known_toss_decisions)
            
            # Enhanced feature engineering
            df_encoded['is_toss_winner_team1'] = (df['toss_winner'] == df['team1']).astype(int)
            df_encoded['is_batting_first'] = (df['toss_decision'] == 'bat').astype(int)
            
            df_encoded['team1_batting_first'] = ((df['toss_winner'] == df['team1']) & 
                                               (df['toss_decision'] == 'bat')).astype(int)
            df_encoded['team2_batting_first'] = ((df['toss_winner'] == df['team2']) & 
                                               (df['toss_decision'] == 'bat')).astype(int)
            
            if is_training:
                df_encoded['target'] = (df['team1'] == df['winner']).astype(int)
            
            features = [
                'team1_encoded', 'team2_encoded',
                'city_encoded', 'venue_encoded',
                'toss_winner_encoded', 'toss_decision_encoded',
                'is_toss_winner_team1', 'is_batting_first',
                'team1_batting_first', 'team2_batting_first'
            ]
            
            if 'target_runs' in df.columns:
                features.extend(['target_runs', 'target_overs'])
            
            X = df_encoded[features]
            
            if is_training:
                self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            if is_training:
                return X_scaled, df_encoded['target']
            return X_scaled
            
        except Exception as e:
            print(f"Error in prepare_features: {str(e)}")
            raise
    
    def optimize_hyperparameters(self, X, y):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        return grid_search.best_estimator_
    
    def train(self, df):
        X, y = self.prepare_features(df, is_training=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model = self.optimize_hyperparameters(X_train, y_train)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, self.model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, self.model.predict_proba(X_test)[:, 1])
        
        # Feature importance
        feature_importance = dict(zip(
            ['team1', 'team2', 'city', 'venue', 'toss_winner', 'toss_decision',
             'is_toss_winner_team1', 'is_batting_first', 'team1_batting_first',
             'team2_batting_first', 'target_runs', 'target_overs'],
            self.model.feature_importances_
        ))
        
        return accuracy, report, cm, fpr, tpr, roc_auc, precision, recall, feature_importance
    
    def plot_confusion_matrix(self, cm):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Team2 Win', 'Team1 Win'], yticklabels=['Team2 Win', 'Team1 Win'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc):
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()
    
    def plot_precision_recall_curve(self, precision, recall):
        plt.figure()
        plt.plot(recall, precision, color='b', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()
    
    def plot_feature_importance(self, feature_importance):
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(10, 6))
        plt.barh(features, importances, color='skyblue')
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Visualization')
        plt.show()

def main():
    df = pd.read_csv(r'C:\Users\manoj\fromnew\SMcric\FEHomePage\output2.csv')
    
    predictor = IPLWinPredictor()
    accuracy, report, cm, fpr, tpr, roc_auc, precision, recall, feature_importance = predictor.train(df)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)
    
    # Plot visualizations
    print("\nConfusion Matrix:")
    predictor.plot_confusion_matrix(cm)
    
    print("\nROC Curve:")
    predictor.plot_roc_curve(fpr, tpr, roc_auc)
    
    print("\nPrecision-Recall Curve:")
    predictor.plot_precision_recall_curve(precision, recall)
    
    print("\nFeature Importance:")
    predictor.plot_feature_importance(feature_importance)
    
    # Example prediction
    match_info = {
        'team1': 'Mumbai Indians',
        'team2': 'Chennai Super Kings',
        'city': 'Mumbai',
        'venue': 'Wankhede Stadium',
        'target_runs': 180,
        'target_overs': 20.0,
        'required_runs': 60,
        'required_wickets': 7,
        'remaining_overs': 5.0,
        'toss_winner': 'Mumbai Indians',
        'toss_decision': 'bat'
    }
    
    try:
        probabilities = predictor.predict_win_probability(match_info)
        print("\nWin Probabilities:")
        print(f"{match_info['team1']}: {probabilities['team1_win_probability']:.2%}")
        print(f"{match_info['team2']}: {probabilities['team2_win_probability']:.2%}")
    except Exception as e:
        print(f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    main()
