(define (problem blocks-006-3)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5)
    (:init (clear b1) (clear b4) (clear b5) (handempty) (on b1 b0) (on b4 b3) (on b5 b2) (ontable b0) (ontable b2) (ontable b3))
    (:goal (and (on b3 b4)))
)