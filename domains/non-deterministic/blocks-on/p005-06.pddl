(define (problem blocks-005-06)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4)
    (:init (clear b0) (clear b3) (handempty) (on b0 b2) (on b2 b4) (on b4 b1) (ontable b1) (ontable b3))
    (:goal (on b2 b0))
)