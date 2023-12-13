(define (problem tireworld-079-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l8 l9 - location)
    (:init (road l0 l21) (road l0 l56) (road l1 l37) (road l1 l4) (road l10 l44) (road l10 l57) (road l10 l6) (road l11 l24) (road l11 l5) (road l12 l46) (road l12 l54) (road l12 l75) (road l13 l21) (road l13 l52) (road l14 l28) (road l14 l53) (road l15 l33) (road l15 l54) (road l16 l25) (road l16 l59) (road l17 l61) (road l17 l68) (road l18 l28) (road l18 l49) (road l19 l40) (road l19 l62) (road l2 l37) (road l2 l39) (road l20 l39) (road l21 l0) (road l21 l13) (road l21 l71) (road l22 l3) (road l22 l58) (road l23 l52) (road l23 l75) (road l24 l11) (road l24 l56) (road l25 l16) (road l25 l44) (road l25 l48) (road l25 l72) (road l26 l70) (road l26 l78) (road l27 l29) (road l27 l77) (road l28 l14) (road l28 l18) (road l29 l27) (road l29 l49) (road l3 l22) (road l3 l66) (road l30 l43) (road l30 l64) (road l31 l55) (road l31 l76) (road l32 l68) (road l32 l69) (road l33 l15) (road l33 l36) (road l34 l35) (road l34 l70) (road l35 l34) (road l35 l41) (road l36 l33) (road l36 l50) (road l37 l1) (road l37 l2) (road l38 l47) (road l38 l53) (road l39 l2) (road l39 l20) (road l4 l1) (road l4 l6) (road l4 l62) (road l4 l72) (road l4 l76) (road l40 l19) (road l40 l43) (road l40 l66) (road l41 l35) (road l41 l67) (road l42 l47) (road l42 l7) (road l43 l30) (road l43 l40) (road l43 l71) (road l44 l10) (road l44 l25) (road l45 l57) (road l45 l75) (road l46 l12) (road l46 l64) (road l47 l38) (road l47 l42) (road l48 l25) (road l48 l7) (road l49 l18) (road l49 l29) (road l5 l11) (road l5 l73) (road l50 l36) (road l50 l51) (road l51 l50) (road l51 l60) (road l52 l13) (road l52 l23) (road l53 l14) (road l53 l38) (road l54 l12) (road l54 l15) (road l55 l31) (road l55 l69) (road l56 l0) (road l56 l24) (road l57 l10) (road l57 l45) (road l58 l22) (road l58 l65) (road l59 l16) (road l59 l8) (road l6 l10) (road l6 l4) (road l60 l51) (road l60 l78) (road l61 l17) (road l61 l74) (road l62 l19) (road l62 l4) (road l62 l65) (road l63 l78) (road l63 l8) (road l64 l30) (road l64 l46) (road l65 l58) (road l65 l62) (road l66 l3) (road l66 l40) (road l67 l41) (road l67 l77) (road l68 l17) (road l68 l32) (road l69 l32) (road l69 l55) (road l7 l42) (road l7 l48) (road l70 l26) (road l70 l34) (road l71 l21) (road l71 l43) (road l72 l25) (road l72 l4) (road l73 l5) (road l73 l9) (road l74 l61) (road l74 l9) (road l75 l12) (road l75 l23) (road l75 l45) (road l76 l31) (road l76 l4) (road l77 l27) (road l77 l67) (road l78 l26) (road l78 l60) (road l78 l63) (road l8 l59) (road l8 l63) (road l9 l73) (road l9 l74) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l8) (spare-in l9) (vehicle-at l20))
    (:goal (vehicle-at l54))
)