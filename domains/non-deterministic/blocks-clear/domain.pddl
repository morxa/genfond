;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 4 Op-blocks world
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain blocksworld)
  (:requirements :strips :typing :non-deterministic)
  (:predicates (on ?x ?y)
	       (ontable ?x)
	       (clear ?x)
	       (handempty)
	       (holding ?x)
	       )

  (:action pickup
	     :parameters (?x)
	     :precondition (and (clear ?x) (ontable ?x) (handempty))
	     :effect
        (oneof (and)
	             (and (not (ontable ?x)) (not (clear ?x)) (not (handempty)) (holding ?x)))
  )

  (:action putdown
	     :parameters (?x)
	     :precondition (holding ?x)
	     :effect
	     (and (not (holding ?x))
		   (clear ?x)
		   (handempty)
		   (ontable ?x)))
  (:action stack
	     :parameters (?x ?y)
	     :precondition (and (holding ?x) (clear ?y))
	     :effect
        (oneof
	        (and (not (holding ?x)) (not (clear ?y)) (clear ?x) (handempty) (on ?x ?y))
          (and (not (holding ?x)) (clear ?x) (handempty) (ontable ?x)))
  )
  (:action unstack
	     :parameters (?x ?y)
	     :precondition (and (on ?x ?y) (clear ?x) (handempty))
	     :effect
        (oneof
	        (and (holding ?x) (clear ?y) (not (clear ?x)) (not (handempty)) (not (on ?x ?y)))
          (and (clear ?y) (ontable ?x) (not (on ?x ?y))))
  )
)
