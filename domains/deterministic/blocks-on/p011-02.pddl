(define (problem blocks-011-02)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b1) (clear b8) (handempty) (on b0 b7) (on b1 b2) (on b10 b9) (on b2 b6) (on b3 b10) (on b4 b5) (on b6 b3) (on b7 b4) (on b9 b0) (ontable b5) (ontable b8))
    (:goal (on b1 b9))
)