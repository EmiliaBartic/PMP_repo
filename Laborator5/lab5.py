from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

cards_game_model = BayesianNetwork(
    [
        ("Card_first_player", "Card_second_player"),
        ("Card_first_player", "P1"),
        ("Card_second_player", "P2"),
        ("P1", "P2"),
        ("Card_first_player", "P3"),
        ("P1", "P3"),
        ("P2", "P3")
    ]
)
# cartea primului jucator cartile de la 1-5 => prob 0.2
CPD_Card_first_player = TabularCPD(variable='Card_first_player', variable_card=5, values=[[0.2], [0.2], [0.2], [0.2], [0.2]])
print('Card_first_player: ',CPD_Card_first_player)

# cartea de al doilea jucator cartile de la 1-5 => prob 0 si 1/4
CPD_Card_second_player = TabularCPD(variable='Card_second_player', variable_card=5, values=[[0, 0.25, 0.25, 0.25, 0.25],
                                                            [0.25, 0, 0.25, 0.25, 0.25], 
                                                            [0.25, 0.25, 0, 0.25, 0.25], 
                                                            [0.25, 0.25, 0.25, 0, 0.25], 
                                                            [0.25, 0.25, 0.25, 0.25, 0]], evidence=['Card_first_player'], evidence_card=[5])
                                                            
print('Card_second_player: ',CPD_Card_second_player)
CPD_P1 = TabularCPD(variable='P1', variable_card=2, values=[[0, 0.2, 0.4, 0.8, 1], [1, 0.8, 0.6, 0.2, 0]], evidence=['Card_first_player'], evidence_card=[5])
# print('P1: ',CPD_P1)

CPD_P2 = TabularCPD(variable='P2', variable_card=3, values=[[0, 0.2, 0.3, 0.4, 1, 0, 0, 0, 0, 0], [1, 0.8, 0.7, 0.6, 0, 1, 0.7, 0.4, 0.3, 0], [0, 0, 0, 0, 0, 0, 0.3, 0.6, 0.7, 1]], evidence=['P1', 'Card_second_player'], evidence_card=[2, 5])
# print('P2: ',CPD_P2)

CPD_P3 = TabularCPD(variable='P3', variable_card=3, values=[[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
                                                            [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
                                                            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]], 
                                evidence=['Card_first_player','P1', 'P2'], evidence_card=[5, 2, 3])
# print('CPD_P3: ',CPD_P3)
cards_game_model.add_cpds(CPD_Card_first_player, CPD_Card_second_player, CPD_P1, CPD_P2, CPD_P3)

cards_game_model.get_cpds()

checking_model = cards_game_model.check_model()
if checking_model == True:
    print('Model ok')
else:
    print('Model not ok')
print('---finish---')

