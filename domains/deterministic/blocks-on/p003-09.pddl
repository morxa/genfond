(define (problem blocks-003-09)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2)
    (:init (clear b0) (clear b1) (clear b2) (handempty) (ontable b0) (ontable b1) (ontable b2))
    (:goal (on b2 b0))
)