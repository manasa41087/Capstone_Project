import numpy as np
import pandas as pd

def batting_statistics(IPL_data):
    #(Player Runs Scored = PRS)
    batsman_runs_scored = IPL_data.groupby(['batsman', 'season'])['batsman_runs'].sum().reset_index()
    #Highest runs scored by each batsman
    batsman_runs_per_match = IPL_data.groupby(['batsman', 'season','match_id'])['batsman_runs'].sum().reset_index()
    highest_score = batsman_runs_per_match.groupby(['batsman','season'])['batsman_runs'].max().reset_index()
    batting_info = pd.merge(batsman_runs_scored,highest_score, on=['batsman','season'], how='inner')
    batting_info.rename(columns={'batsman_runs_x':'batsman_runs_scored','batsman_runs_y':'highest_score'},inplace=True)
    #(BFP) = (batsman_balls_faced) - (batsman_wide_faced)
    batsman_balls_faced = IPL_data.groupby(['batsman', 'season'])['over'].count().reset_index()
    batsman_wide_faced = IPL_data.groupby(['batsman', 'season'])['wide_runs'].sum().reset_index()
    #(NTO = Number of times out)
    batsman_wicket = IPL_data.groupby(['batsman', 'season'])['dismissal_kind'].count().reset_index()
    #(TRS =  Tournament runs scored)
    tournament_runs_scored = IPL_data.groupby('season')['batsman_runs'].sum().reset_index()
    #(BFT = Balls faced) ->(tournament_balls_faced - tournament_wide_faced)
    tournament_balls_faced = IPL_data.groupby(['season'])['over'].count().reset_index()
    tournament_wide_faced = IPL_data.groupby(['season'])['wide_runs'].sum().reset_index()
    #(TWF = Tournament wicket fallen)
    tournament_batsman_wicket = IPL_data.groupby(['season'])['dismissal_kind'].count().reset_index()
    #Tournament runs conceded (TRC1+TRC2)
    TRC1 = IPL_data.groupby('season')['batsman_runs'].sum().reset_index()
    TRC2 = IPL_data.groupby(['season'])['wide_runs'].sum().reset_index()
    #Number of fours and sixes
    number_of_fours_df = IPL_data[IPL_data['batsman_runs'] == 4]
    number_of_fours = number_of_fours_df.groupby(['batsman','season'])['batsman_runs'].count().reset_index()
    number_of_fours.rename(columns={'batsman_runs':'number_of_fours'}, inplace=True)
    number_of_sixes_df = IPL_data[IPL_data['batsman_runs'] == 6]
    number_of_sixes = number_of_sixes_df.groupby(['batsman','season'])['batsman_runs'].count().reset_index()
    number_of_sixes.rename(columns={'batsman_runs':'number_of_sixes'}, inplace=True)
    #Number of centuries and half centuries
    batsman_runs_match_id = IPL_data.groupby(['batsman', 'match_id','season'])['batsman_runs'].sum().reset_index()
    all_half_centuries = batsman_runs_match_id[batsman_runs_match_id['batsman_runs'] >= 50]
    number_half_centuries = all_half_centuries.groupby(['batsman','season'])['match_id'].count().reset_index()
    all_centuries = batsman_runs_match_id[batsman_runs_match_id['batsman_runs'] >= 100]
    number_of_centuries = all_centuries.groupby(['batsman','season'])['match_id'].count().reset_index()
    # Merging them to be a dataframe
    batting_info = pd.merge(batting_info,batsman_balls_faced, on=['batsman','season'], how='inner')
    batting_info = pd.merge(batting_info,batsman_wide_faced, on=['batsman','season'], how='inner')
    batting_info = pd.merge(batting_info, batsman_wicket, on=['batsman','season'], how='inner')
    batting_info = pd.merge(batting_info,tournament_runs_scored, on=['season'], how='left')
    batting_info = pd.merge(batting_info,tournament_balls_faced, on=['season'], how='left')
    batting_info = pd.merge(batting_info,tournament_wide_faced, on=['season'], how='left')
    batting_info = pd.merge(batting_info,tournament_batsman_wicket, on=['season'], how='left')
    batting_info = pd.merge(batting_info,TRC1, on=['season'], how='left')
    batting_info = pd.merge(batting_info,TRC2, on=['season'], how='left')
    batting_info = pd.merge(batting_info, number_of_fours, on=['batsman','season'], how='left')
    batting_info = pd.merge(batting_info, number_of_sixes, on=['batsman','season'], how='left')
    batting_info = pd.merge(batting_info, number_half_centuries, on=['batsman','season'], how='left')
    batting_info = pd.merge(batting_info, number_of_centuries, on=['batsman','season'], how='left')
    #renaming the columns
    batting_info.rename(columns={'over_x': 'batsman_balls_faced','wide_runs_x':'batsman_wide_faced',
    'dismissal_kind_x': 'batsman_wicket','batsman_runs_x':'tournament_runs_scored','over_y':'tournament_balls_faced','wide_runs_y':'tournament_wide_faced',
    'dismissal_kind_y':'tournament_batsman_wicket','batsman_runs_y':'TRC1','wide_runs':'TRC2','match_id_x':'number_half_centuries',
    'match_id_y':'number_of_centuries'}, inplace=True)
    #batting statistics
    batting_info['balls_faced'] = batting_info['batsman_balls_faced'] - batting_info['batsman_wide_faced']
    batting_info['tournament_b_faced'] = batting_info['tournament_balls_faced'] - batting_info['tournament_wide_faced']
    batting_info['batting_strike_rate'] = ((batting_info['batsman_runs_scored']) * 100) / (batting_info['balls_faced'])
    #making the batsman wicket from 0 to 1 to avoid infinity
    batting_info.batsman_wicket.replace(to_replace=0,value=1, inplace=True)
    batting_info['player_batting_average'] = (batting_info['batsman_runs_scored']) / (batting_info['batsman_wicket'])
    batting_info['tournament_batting_strike_rate'] = batting_info['tournament_runs_scored'] * 100 / (batting_info['tournament_b_faced'])
    batting_info['tournament_batting_average'] = batting_info['tournament_runs_scored'] / (batting_info['tournament_batsman_wicket'])
    batting_info['batting_index'] = batting_info['player_batting_average'] * batting_info['batting_strike_rate']
    #fill null values
    batting_info.number_of_fours.fillna(0,inplace=True)
    batting_info.number_of_sixes.fillna(0,inplace=True)
    batting_info.number_half_centuries.fillna(0,inplace=True)
    batting_info.number_of_centuries.fillna(0,inplace=True)
    #batting_info.player_batting_average.fillna(0,inplace=True)
    #Bowling statistics for merging which can be droppedlater

    batting_info['TRC'] = batting_info['TRC1'] + batting_info['TRC2']
    batting_info['TBA'] = batting_info['tournament_b_faced'] / batting_info['tournament_batsman_wicket']
    batting_info['TBER'] = (batting_info['TRC'] * 6) / batting_info['tournament_b_faced']
    batting_info['TBSR'] = batting_info['TRC'] / batting_info['tournament_batsman_wicket']
    #batting_info = batting_info.replace([np.inf,-np.inf],np.nan)
    #batting_info.fillna(0,inplace=True)
    return batting_info

def bowling_statistics(IPL_data):
    #(PRC= Player runs conceded) = bowler_runs_given + bowler_wide_given
    bowler_runs_given = IPL_data.groupby(['bowler', 'season'])['batsman_runs'].sum().reset_index()
    bowler_wide_given = IPL_data.groupby(['bowler', 'season'])['wide_runs'].sum().reset_index()
    #(PBB)
    bowler_balls_bowled = IPL_data.groupby(['bowler', 'season'])['over'].count().reset_index()
    #(TBB)
    tournament_balls_bowled = IPL_data.groupby('season')['over'].count().reset_index()
    #(TRC)
    #Tournament runs conceded (TRC1+TRC2)
    TRC1 = IPL_data.groupby('season')['batsman_runs'].sum().reset_index()
    TRC2 = IPL_data.groupby(['season'])['wide_runs'].sum().reset_index()
    #tournament_runs_conceded = IPL_data.groupby('season')['total_runs'].sum().reset_index()
    #(PWT)
    player_wickets_taken = IPL_data.groupby(['bowler', 'season'])['dismissal_kind'].count().reset_index()
    #(TWT)
    tournament_wickets_taken = IPL_data.groupby('season')['dismissal_kind'].count().reset_index()
    #getting it all together
    bowling_info = pd.merge(bowler_runs_given, bowler_balls_bowled, on=['bowler','season'], how='inner')
    bowling_info = pd.merge(bowling_info, bowler_wide_given, on=['bowler','season'], how='inner')
    bowling_info = pd.merge(bowling_info, player_wickets_taken, on=['bowler','season'], how='inner')
    bowling_info = pd.merge(bowling_info, tournament_balls_bowled, on=['season'], how='left')
    bowling_info = pd.merge(bowling_info, TRC1, on=['season'], how='left')
    bowling_info = pd.merge(bowling_info, TRC2, on=['season'], how='left')
    bowling_info = pd.merge(bowling_info, tournament_wickets_taken, on=['season'], how='left')
    #renaming columns
    bowling_info.rename(columns={'batsman_runs_x':'bowler_runs_given','over_x':'bowler_balls_bowled','wide_runs_x':'bowler_wide_given',
    'dismissal_kind_x':'player_wickets_taken','over_y':'tournament_balls_bowled','batsman_runs_y': 'TRC1','wide_runs_y': 'TRC2','dismissal_kind_y': 'tournament_wickets_taken'}, inplace=True)
    #making the batsman wicket from 0 to 1 to avoid infinity
    bowling_info.player_wickets_taken.replace(to_replace=0,value=1, inplace=True)
    #bowling stats
    bowling_info['player_runs_conceded'] = bowling_info['bowler_runs_given'] + bowling_info['bowler_wide_given']
    bowling_info['tournament_runs_conceded'] = bowling_info['TRC1'] + bowling_info['TRC2']
    bowling_info['player_bowling_average'] = bowling_info['player_runs_conceded'] / bowling_info['player_wickets_taken']
    bowling_info['tournament_bowling_average'] = bowling_info['tournament_runs_conceded'] / bowling_info['tournament_wickets_taken']
    bowling_info['player_bowling_economy_rate'] = (bowling_info['player_runs_conceded'] * 6) / bowling_info['bowler_balls_bowled']
    bowling_info['tournament_bowling_economy_rate'] = (bowling_info['tournament_runs_conceded'] * 6) / bowling_info['tournament_balls_bowled']
    bowling_info['player_bowling_strike_rate'] = bowling_info['bowler_balls_bowled'] / bowling_info['player_wickets_taken']
    bowling_info['tournament_bowler_strike_rate'] = bowling_info['tournament_balls_bowled'] / bowling_info['tournament_wickets_taken']
    bowling_info['bowling_index'] = bowling_info['player_bowling_average'] * bowling_info['player_bowling_economy_rate']
    #fill infinity values eith NaN
    #bowling_info = bowling_info.replace([np.inf,-np.inf],np.nan)
    #bowling_info.fillna(0,inplace=True)
    return bowling_info

def with_auction_data(batting_info, bowling_info, auction_original):
    #batting_info with Salary
    batting_auction_data = pd.merge(batting_info, auction_original, left_on=['batsman','season'], right_on=['Player','season'], how='left')
    #bowling_info with Salary
    bowling_auction_data = pd.merge(bowling_info, auction_original, left_on=['bowler','season'], right_on=['Player','season'], how='left')
    #batting and bowling info merged with Salary
    overall_statistics = pd.merge(batting_auction_data, bowling_auction_data, left_on=['batsman','season'], right_on=['bowler','season'], how='outer')
    #overall_statistics_auction_data = pd.merge(overall_statistics, auction_calc_data, left_on=['batsman','season'], right_on=['Player','season'], how='outer')

    '''batting Nan values'''
    overall_statistics.batsman.fillna(overall_statistics.bowler,inplace=True)
    overall_statistics.batsman_runs_scored.fillna(-99999,inplace=True)
    overall_statistics.highest_score.fillna(-99999,inplace=True)
    overall_statistics.batsman_balls_faced.fillna(-99999,inplace=True)
    overall_statistics.batsman_wide_faced.fillna(-999999,inplace=True)
    overall_statistics.batsman_wicket.fillna(-99999,inplace=True)
    overall_statistics.number_of_fours.fillna(-999999,inplace=True)
    overall_statistics.number_of_sixes.fillna(-99999,inplace=True)
    overall_statistics.number_half_centuries.fillna(-99999,inplace=True)
    overall_statistics.number_of_centuries.fillna(-99999,inplace=True)
    overall_statistics.balls_faced.fillna(-99999,inplace=True)
    overall_statistics.batting_strike_rate.fillna(-99999,inplace=True)
    overall_statistics.player_batting_average.fillna(-99999,inplace=True)
    overall_statistics.batting_index.fillna(-99999,inplace=True)

    #tournament stuff
    overall_statistics.tournament_runs_scored.fillna((overall_statistics['tournament_runs_scored'].mean()),inplace=True)
    overall_statistics.tournament_balls_faced.fillna(overall_statistics.tournament_balls_bowled,inplace=True)
    overall_statistics.tournament_batsman_wicket.fillna((overall_statistics['tournament_batsman_wicket'].mean()),inplace=True)
    overall_statistics.tournament_b_faced.fillna(overall_statistics.tournament_balls_bowled,inplace=True)
    overall_statistics.tournament_batting_strike_rate.fillna((overall_statistics['tournament_batting_strike_rate'].mean()),inplace=True)
    overall_statistics.tournament_batting_average.fillna((overall_statistics['tournament_batting_average'].mean()),inplace=True)
    overall_statistics.TRC1_x.fillna(overall_statistics.TRC1_y,inplace=True)
    overall_statistics.TRC2_x.fillna(overall_statistics.TRC2_y,inplace=True)
    overall_statistics.TRC.fillna(overall_statistics.tournament_runs_conceded,inplace=True)
    overall_statistics.TBA.fillna(overall_statistics.tournament_bowling_average,inplace=True)
    overall_statistics.TBER.fillna(overall_statistics.tournament_bowling_economy_rate,inplace=True)
    overall_statistics.TBSR.fillna(overall_statistics.tournament_bowler_strike_rate,inplace=True)

    '''Nan values bowling'''
    overall_statistics.bowler_runs_given.fillna(-99999,inplace=True)
    overall_statistics.bowler_balls_bowled.fillna(-99999,inplace=True)
    overall_statistics.bowler_wide_given.fillna(-99999,inplace=True)
    overall_statistics.player_wickets_taken.fillna(-99999,inplace=True)
    overall_statistics.tournament_balls_bowled.fillna(overall_statistics.tournament_balls_faced,inplace=True)
    overall_statistics.TRC1_y.fillna(overall_statistics.TRC1_x,inplace=True)
    overall_statistics.TRC2_y.fillna(overall_statistics.TRC2_x,inplace=True)
    overall_statistics.tournament_wickets_taken.fillna(overall_statistics.tournament_batsman_wicket,inplace=True)
    overall_statistics.player_runs_conceded.fillna(-99999,inplace=True)

    overall_statistics.player_bowling_strike_rate.fillna(-99999,inplace=True)
    overall_statistics.player_bowling_average.fillna(-99999,inplace=True)
    overall_statistics.player_bowling_economy_rate.fillna(-99999,inplace=True)
    overall_statistics.bowling_index.fillna(-99999,inplace=True)
    overall_statistics.tournament_runs_conceded.fillna(overall_statistics.TRC,inplace=True)
    overall_statistics.tournament_bowling_average.fillna(overall_statistics.TBA,inplace=True)
    overall_statistics.tournament_bowling_economy_rate.fillna(overall_statistics.TBER,inplace=True)
    overall_statistics.tournament_bowler_strike_rate.fillna(overall_statistics.TBSR,inplace=True)
    overall_statistics.tournament_balls_faced.fillna(overall_statistics.tournament_balls_bowled,inplace=True)
    overall_statistics.tournament_batsman_wicket.fillna(overall_statistics.tournament_wickets_taken,inplace=True)

    '''Working with overall_statistics'''
    overall_statistics.Salary_y.fillna(overall_statistics.Salary_x,inplace=True)
    overall_statistics.rename(columns={'Salary_y': 'Salary', 'batsman':'Player'},inplace=True)
    overall_statistics.drop(['tournament_wide_faced','Player_x', 'Player_y','Salary_x','bowler'], axis=1, inplace=True)
    batting_auction_data.drop(['Player'],axis=1,inplace=True)
    return batting_auction_data, bowling_auction_data, overall_statistics


def load_overall_statistics_x_y():
    '''
    INPUT
         - none
    OUTPUT
         - overall statistics: pandas dataframe of the features
         - X: 2d array, features
         - y: 1d array, the target
    Returns overall_statistics, X, y after executing all the functions above
    '''
    matches = pd.read_csv("data/matches.csv")
    deliveries = pd.read_csv("data/deliveries.csv")
    IPL_data = pd.merge(deliveries, matches, left_on='match_id', right_on='id', how='left')
    batting_info = batting_statistics(IPL_data)
    bowling_info = bowling_statistics(IPL_data)
    auction_original = pd.read_csv("data/auction_original_data.csv")
    auction_original = auction_original[['Player','season','Salary']]
    auction_original['Salary'] = auction_original['Salary'].str.replace('$', '')
    auction_original['Salary'] = auction_original['Salary'].str.replace(',', '')
    auction_original['Salary'] = auction_original.Salary.apply(float)
    batting_auction_data, bowling_auction_data, overall_statistics = with_auction_data(batting_info, bowling_info, auction_original)
    #overall_statistics.fillna(-99999,inplace=True)
    #overall_statistics.drop(['Salary_x', 'bowler','TRC','TBA','TBER','TBSR'],axis=1,inplace=True)
    #overall_statistics.rename(columns={'Salary_y': 'Salary', 'batsman':'Player'},inplace=True)
    overall_statistics = overall_statistics.dropna(how='any')
    batting_auction_data = batting_auction_data.dropna(how='any')
    y = overall_statistics['Salary']
    X = overall_statistics.drop(['Player','Salary'],axis=1)
    return overall_statistics, X, y, batting_auction_data, bowling_auction_data

if __name__ == '__main__':
    matches = pd.read_csv("data/matches.csv")
    deliveries = pd.read_csv("data/deliveries.csv")
    IPL_data = pd.merge(deliveries, matches, left_on='match_id', right_on='id', how='left')
    batting_info = batting_statistics(IPL_data)
    bowling_info = bowling_statistics(IPL_data)
    auction_original = pd.read_csv("data/auction_original_data.csv")
    auction_original = auction_original[['Player','season','Salary']]
    batting_auction_data, bowling_auction_data, overall_statistics = with_auction_data(batting_info, bowling_info, auction_original)
    overall_statistics.fillna(-99999,inplace=True)
    overall_statistics.drop(['Salary_x', 'bowler','TRC','TBA','TBER','TBSR'],axis=1,inplace=True)
    overall_statistics.rename(columns={'Salary_y': 'Salary', 'batsman':'Player'},inplace=True)

    overall_statistics['Salary'] = overall_statistics['Salary'].str.replace('$', '')
    overall_statistics['Salary'] = overall_statistics['Salary'].str.replace(',', '')
    overall_statistics['Salary'] = overall_statistics.Salary.apply(float)
    overall_statistics = overall_statistics.dropna(how='any')
    y = overall_statistics['Salary']
    X = overall_statistics.drop(['Player','Salary'],axis=1)
