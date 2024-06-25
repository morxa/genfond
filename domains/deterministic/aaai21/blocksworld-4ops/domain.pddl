(define (domain blocksworld)
    (:requirements :strips)
    (:predicates (arm-empty) (clear ?x)  (holding ?x)  (on ?x ?y)  (on-table ?x))
    (:action pickup
        :parameters (?ob)
        :precondition (and (clear ?ob) (on-table ?ob) (arm-empty))
        :effect (and (holding ?ob) (not (clear ?ob)) (not (on-table ?ob)) (not (arm-empty)))
    )
     (:action putdown
        :parameters (?ob)
        :precondition (holding ?ob)
        :effect (and (clear ?ob) (arm-empty) (on-table ?ob) (not (holding ?ob)))
    )
     (:action stack
        :parameters (?ob ?underob)
        :precondition (and (clear ?underob) (holding ?ob))
        :effect (and (arm-empty) (clear ?ob) (on ?ob ?underob) (not (clear ?underob)) (not (holding ?ob)))
    )
     (:action unstack
        :parameters (?ob ?underob)
        :precondition (and (on ?ob ?underob) (clear ?ob) (arm-empty))
        :effect (and (holding ?ob) (clear ?underob) (not (on ?ob ?underob)) (not (clear ?ob)) (not (arm-empty)))
    )
)