(define (problem blocks-003-05)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2)
    (:init (clear b1) (handempty) (on b1 b2) (on b2 b0) (ontable b0))
    (:goal (on b2 b1))
)