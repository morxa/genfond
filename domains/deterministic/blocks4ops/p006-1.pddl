(define (problem blocks-006-1)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5)
    (:init (clear b1) (clear b4) (handempty) (on b1 b2) (on b2 b5) (on b4 b0) (on b5 b3) (ontable b0) (ontable b3))
    (:goal (and (on b1 b3) (on b3 b2)))
)