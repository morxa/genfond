(define (problem blocks-013-02)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b12) (clear b3) (clear b9) (handempty) (on b0 b8) (on b1 b7) (on b11 b10) (on b12 b6) (on b2 b4) (on b4 b1) (on b6 b11) (on b7 b5) (on b8 b2) (ontable b10) (ontable b3) (ontable b5) (ontable b9))
    (:goal (on b0 b1))
)