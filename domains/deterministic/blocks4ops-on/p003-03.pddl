(define (problem blocks-003-03)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2)
    (:init (clear b1) (clear b2) (handempty) (on b1 b0) (ontable b0) (ontable b2))
    (:goal (on b0 b2))
)