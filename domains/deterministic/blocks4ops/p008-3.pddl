(define (problem blocks-008-3)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7)
    (:init (clear b7) (handempty) (on b0 b5) (on b1 b3) (on b3 b0) (on b4 b1) (on b5 b2) (on b6 b4) (on b7 b6) (ontable b2))
    (:goal (and (on b2 b3) (on b3 b1) (on b1 b7)))
)