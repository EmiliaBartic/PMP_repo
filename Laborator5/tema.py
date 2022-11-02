import pgmpy as pg
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

#EX1

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

model_inferential = VariableElimination(cards_game_model)
# if checking_model == True:
#     print('Model ok')
# else:
#     print('Model not ok')
# print('---finish---')

#EX 2 
# a - esti jucatorul 1 si ai primit un rege de frunza, ar trebui sa pariezi sau nu?

query1 = model_inferential.query(variables=['P1'], evidence={'Card_first_player': 1})
print('first query 1:' ,query1)

#b - esti jucatorul 2 si ai primit un rege de inima iar jucatorul 1 a decis sa parieze , cum ar trb sa decizi?
query2 = model_inferential.query(variables=['P2'], evidence={'P1': 0, 'Card_second_player': 2})
print('first query 2:',query2)

# c - schimbati parametrii strategiilor de pariere - schimb P1 si P2

P1_modified = TabularCPD(variable = 'P1', variable_card = 2,values = [[1, 0.2, 0.6, 0.3, 0],[0, 0.8, 0.4, 0.7, 1]], evidence = ['Card_first_player'],evidence_card=[5])

P2_modified = TabularCPD(variable = 'P2', variable_card=3, values=[[1, 0.8, 0.4, 0.3, 0, 1, 0.8, 0.4, 0.2, 0],[0, 0.2, 0.6, 0.7, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0.2, 0.6, 0.8, 1]], evidence=['P1', 'Card_second_player'], evidence_card=[2, 5])

print(P1_modified, P2_modified)

# # recalculare queries a,b

query1_modified = model_inferential.query(variables=['P1'], evidence={'Card_first_player': 1})
print('second query 1:' ,query1_modified)

#b - esti jucatorul 2 si ai primit un rege de inima iar jucatorul 1 a decis sa parieze , cum ar trb sa decizi?
query2_modified = model_inferential.query(variables=['P2'], evidence={'P1': 0, 'Card_second_player': 2})
print('second query 2:',query2_modified)
