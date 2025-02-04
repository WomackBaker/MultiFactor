import owlready2
from owlready2 import *
onto = get_ontology("http://test.org/drug.owl")

with onto:
    class id_5b95bc47726e4ec58c1d58e25e3c0f45(Thing): pass

    class id_5b95bc47726e4ec58c1d58e25e3c0f451(Thing): pass
    class latitude(id_5b95bc47726e4ec58c1d58e25e3c0f451 >> int, DataProperty): pass
    class longitude(id_5b95bc47726e4ec58c1d58e25e3c0f451 >> int, DataProperty): pass

    class attacker(id_5b95bc47726e4ec58c1d58e25e3c0f451 >> bool, DataProperty): pass


    rule = Imp()
    rule.set_as_rule("""id_5b95bc47726e4ec58c1d58e25e3c0f451(?u) ^ longitude(?u,?l) ^ notEqual(?l, 5) -> attacker(?u, true)""")

    user1 = id_5b95bc47726e4ec58c1d58e25e3c0f451()
    user1.latitude.append(10)

    user2 = id_5b95bc47726e4ec58c1d58e25e3c0f451()
    user2.longitude.append(20)

sync_reasoner_pellet(infer_property_values = True, infer_data_property_values = True)
print(user2.attacker)

