(define (domain gripper-strips)
    (:predicates (at ?b ?r)  (at-robby ?r)  (ball ?b)  (carry ?o ?g)  (free ?g)  (gripper ?g)  (room ?r))
    (:action drop
        :parameters (?obj ?room ?gripper)
        :precondition (and (ball ?obj) (room ?room) (gripper ?gripper) (carry ?obj ?gripper) (at-robby ?room))
        :effect (and (at ?obj ?room) (free ?gripper) (not (carry ?obj ?gripper)))
    )
     (:action move
        :parameters (?from ?to)
        :precondition (and (room ?from) (room ?to) (at-robby ?from))
        :effect (and (at-robby ?to) (not (at-robby ?from)))
    )
     (:action pick
        :parameters (?obj ?room ?gripper)
        :precondition (and (ball ?obj) (room ?room) (gripper ?gripper) (at ?obj ?room) (at-robby ?room) (free ?gripper))
        :effect (and (carry ?obj ?gripper) (not (at ?obj ?room)) (not (free ?gripper)))
    )
)