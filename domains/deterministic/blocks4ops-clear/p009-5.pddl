(define (problem blocks-009-5)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8)
    (:init (clear b5) (clear b7) (handempty) (on b0 b8) (on b3 b0) (on b4 b3) (on b5 b4) (on b6 b1) (on b7 b2) (on b8 b6) (ontable b1) (ontable b2))
    (:goal (clear b3))
)