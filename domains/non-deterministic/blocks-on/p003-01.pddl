(define (problem blocks-003-01)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2)
    (:init (clear b2) (handempty) (on b1 b0) (on b2 b1) (ontable b0))
    (:goal (on b0 b1))
)