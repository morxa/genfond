;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 3 Op-blocks world
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain blocks2)
  (:requirements :strips)
  (:predicates (on ?x ?y)
	       (ontable ?x)
	       (clear ?x)
               (neq ?x ?y)
	       )

  (:action newtower
    :parameters (?x ?y)
    :precondition (and (clear ?x) (on ?x ?y))
    :effect (and (ontable ?x) (clear ?y) (not (on ?x ?y))))

  (:action stack
    :parameters (?x ?y)
    :precondition (and (neq ?x ?y) (clear ?x) (clear ?y) (ontable ?x))
    :effect (and (on ?x ?y) (not (ontable ?x)) (not (clear ?y))))

  (:action move
    :parameters (?x ?y ?z)
    :precondition (and (neq ?x ?z) (clear ?x) (clear ?z) (on ?x ?y))
    :effect (and (on ?x ?z) (not (clear ?z)) (clear ?y) (not (on ?x ?y))))
)

