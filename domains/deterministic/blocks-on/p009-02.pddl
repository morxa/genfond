(define (problem blocks-009-02)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8)
    (:init (clear b1) (clear b4) (handempty) (on b1 b2) (on b2 b6) (on b4 b8) (on b5 b3) (on b6 b5) (on b7 b0) (on b8 b7) (ontable b0) (ontable b3))
    (:goal (on b3 b7))
)