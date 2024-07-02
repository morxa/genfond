(define (domain miconic)
    (:requirements :strips :typing)
    (:types floor passenger - object)
    (:predicates (above ?floor1 - floor ?floor2 - floor)  (boarded ?person - passenger)  (destin ?person - passenger ?floor - floor)  (lift-at ?floor - floor)  (not-boarded ?person - passenger)  (not-served ?person - passenger)  (origin ?person - passenger ?floor - floor)  (served ?person - passenger))
    (:action board
        :parameters (?f - floor ?p - passenger)
        :precondition (and (lift-at ?f) (origin ?p ?f))
        :effect (boarded ?p)
    )
     (:action depart
        :parameters (?f - floor ?p - passenger)
        :precondition (and (lift-at ?f) (destin ?p ?f) (boarded ?p))
        :effect (and (not (boarded ?p)) (served ?p))
    )
     (:action down
        :parameters (?f1 - floor ?f2 - floor)
        :precondition (and (lift-at ?f1) (above ?f2 ?f1))
        :effect (and (lift-at ?f2) (not (lift-at ?f1)))
    )
     (:action up
        :parameters (?f1 - floor ?f2 - floor)
        :precondition (and (lift-at ?f1) (above ?f1 ?f2))
        :effect (and (lift-at ?f2) (not (lift-at ?f1)))
    )
)