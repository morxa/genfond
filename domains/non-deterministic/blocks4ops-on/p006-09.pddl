(define (problem blocks-006-09)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5)
    (:init (clear b4) (clear b5) (handempty) (on b1 b0) (on b2 b1) (on b4 b3) (on b5 b2) (ontable b0) (ontable b3))
    (:goal (on b3 b1))
)