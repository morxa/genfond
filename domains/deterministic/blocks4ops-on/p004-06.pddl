(define (problem blocks-004-06)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3)
    (:init (clear b3) (handempty) (on b1 b0) (on b2 b1) (on b3 b2) (ontable b0))
    (:goal (on b3 b2))
)